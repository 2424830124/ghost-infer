//! Next-token selection: greedy, or temperature + top-k + top-p sampling; optionally penalties against
//! repeating (a presence penalty on words already said, and DRY on runs of tokens already said).

use std::collections::{HashMap, HashSet};

/// DRY's growth per token a run is longer, the run it lets pass, and the longest it counts (as llama.cpp's
/// defaults).
const DRY_BASE: f32 = 1.75;
const DRY_ALLOWED: usize = 2;
const DRY_MAX_RUN: usize = 48;

pub struct Sampler {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    rng: u64,
    /// a presence penalty: taken off the logit of every word already said (in recent replies, and in this one);
    /// `word` says which tokens count as words
    presence: f32,
    seen: HashSet<u32>,
    word: Option<Box<dyn Fn(u32) -> bool + Send>>,
    /// DRY ("don't repeat yourself"): a token that would carry on a run of tokens the history already has
    /// is held back by dry x DRY_BASE^(run - DRY_ALLOWED), so a phrase may come back but a whole earlier
    /// sentence cannot be built again. `breaker`: tokens a run does not cross and that are never held back
    /// (line breaks, special tokens such as the reply's end)
    dry: f32,
    history: Vec<u32>,
    breaker: Option<Box<dyn Fn(u32) -> bool + Send>>,
    /// this step's penalties, by token
    held: HashMap<u32, f32>,
    /// a repetition penalty as Hugging Face's and llama.cpp's: a token already in the context or the reply has
    /// its logit divided by it (multiplied, when negative). Audio-token models (maya1, Orpheus) loop without it.
    repetition: f32,
    used: HashSet<u32>,
    /// tokens never picked
    banned: HashSet<u32>,
}

impl Sampler {
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Sampler {
        Sampler { temperature, top_k, top_p, rng: seed.max(1), presence: 0.0, seen: HashSet::new(), word: None, dry: 0.0, history: Vec::new(), breaker: None, held: HashMap::new(), repetition: 1.0, used: HashSet::new(), banned: HashSet::new() }
    }

    /// Never picking any of `ids`.
    pub fn with_ban(mut self, ids: impl IntoIterator<Item = u32>) -> Sampler {
        self.banned.extend(ids);
        self
    }

    /// With a repetition penalty of `penalty` (above 1) on the tokens of `context` and those it picks from now on.
    pub fn with_repetition(mut self, penalty: f32, context: impl IntoIterator<Item = u32>) -> Sampler {
        self.repetition = penalty.max(1.0);
        self.used = context.into_iter().collect();
        self
    }

    /// With a presence penalty of `penalty` on the words among `said` and on those it picks from now on.
    pub fn with_presence(mut self, penalty: f32, said: impl IntoIterator<Item = u32>, word: impl Fn(u32) -> bool + Send + 'static) -> Sampler {
        self.seen = said.into_iter().filter(|&t| word(t)).collect();
        self.word = Some(Box::new(word));
        self.presence = penalty.max(0.0);
        self
    }

    /// With DRY at `multiplier` over `history` (the context's last tokens) and what it picks from now on.
    pub fn with_dry(mut self, multiplier: f32, history: Vec<u32>, breaker: impl Fn(u32) -> bool + Send + 'static) -> Sampler {
        self.dry = multiplier.max(0.0);
        self.history = history;
        self.breaker = Some(Box::new(breaker));
        self
    }

    /// A picked token, remembered for the penalties.
    fn note(&mut self, id: u32) {
        if self.presence > 0.0 && self.word.as_ref().is_some_and(|w| w(id)) {
            self.seen.insert(id);
        }
        if self.dry > 0.0 {
            self.history.push(id);
        }
        if self.repetition > 1.0 {
            self.used.insert(id);
        }
    }

    /// This step's penalties: the presence penalty's, and DRY's for each token that would carry on a run the
    /// history repeats (the longest such run for that token).
    fn weigh(&mut self) {
        self.held.clear();
        if self.presence > 0.0 {
            for &t in &self.seen {
                self.held.insert(t, self.presence);
            }
        }
        let (h, n) = (&self.history, self.history.len());
        let Some(brk) = self.breaker.as_ref().filter(|_| self.dry > 0.0 && n >= 2) else { return };
        let last = h[n - 1];
        if brk(last) {
            return;
        }
        let mut dry: HashMap<u32, usize> = HashMap::new();
        for j in 0..n - 1 {
            if h[j] != last {
                continue;
            }
            // h[..=j] ends as the history does, for `run` tokens; the token after it would make the run longer
            let mut run = 1;
            while run < DRY_MAX_RUN && run <= j && h[j - run] == h[n - 1 - run] && !brk(h[j - run]) {
                run += 1;
            }
            let next = h[j + 1];
            if run >= DRY_ALLOWED && !brk(next) {
                let e = dry.entry(next).or_insert(0);
                *e = (*e).max(run);
            }
        }
        for (t, run) in dry {
            *self.held.entry(t).or_insert(0.0) += self.dry * DRY_BASE.powi((run - DRY_ALLOWED) as i32);
        }
    }

    /// A token's logit, less this step's penalty for it.
    fn score(&self, id: usize, v: f32) -> f32 {
        if !self.banned.is_empty() && self.banned.contains(&(id as u32)) {
            return f32::NEG_INFINITY;
        }
        let v = if self.repetition > 1.0 && self.used.contains(&(id as u32)) {
            if v > 0.0 { v / self.repetition } else { v * self.repetition }
        } else {
            v
        };
        if self.held.is_empty() { v } else { v - self.held.get(&(id as u32)).copied().unwrap_or(0.0) }
    }

    /// The most likely token, after the penalty.
    fn best(&self, logits: &[f32]) -> u32 {
        logits.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &v)| {
            let v = self.score(i, v);
            if v > b.1 { (i, v) } else { b }
        }).0 as u32
    }

    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        ((self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / (1u64 << 24) as f32
    }

    /// Temperature 0 and no penalty: always the most likely token (which the GPU can pick on its own).
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0 && self.presence <= 0.0 && self.dry <= 0.0 && self.repetition <= 1.0 && self.banned.is_empty()
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        self.weigh();
        let id = self.pick(logits);
        self.note(id);
        id
    }

    fn pick(&mut self, logits: &[f32]) -> u32 {
        if self.temperature <= 0.0 {
            return self.best(logits);
        }
        let (cand, mass) = self.distribution(logits);
        let mut r = self.next_f32() * mass;
        for &(id, p) in &cand {
            r -= p;
            if r <= 0.0 {
                return id;
            }
        }
        cand.last().map(|c| c.0).unwrap_or_else(|| self.best(logits))
    }

    /// Whether to keep `draft`, a token proposed for the position these logits score: Ok, or Err with the
    /// token to take instead. Kept with the probability p(draft) that sampling gives it; otherwise the token
    /// is drawn from the distribution with `draft` left out. Either way the result is distributed exactly as
    /// `sample` would pick (speculative sampling with a draft that is sure of its one token).
    pub fn accept(&mut self, logits: &[f32], draft: u32) -> Result<(), u32> {
        self.weigh();
        let r = self.judge(logits, draft);
        self.note(match r {
            Ok(()) => draft,
            Err(t) => t,
        });
        r
    }

    fn judge(&mut self, logits: &[f32], draft: u32) -> Result<(), u32> {
        if self.temperature <= 0.0 {
            let best = self.best(logits);
            return if best == draft { Ok(()) } else { Err(best) };
        }
        let (cand, mass) = self.distribution(logits);
        let p = cand.iter().find(|c| c.0 == draft).map_or(0.0, |c| c.1);
        if self.next_f32() * mass < p {
            return Ok(());
        }
        let mut r = self.next_f32() * (mass - p);
        let mut fallback = None;
        for &(id, q) in cand.iter().filter(|c| c.0 != draft) {
            fallback = Some(id);
            r -= q;
            if r <= 0.0 {
                return Err(id);
            }
        }
        // rounding left a sliver over: the last token, or the draft when it was the only candidate
        fallback.map_or(Ok(()), Err)
    }

    /// The tokens sampling can pick, most likely first, with their weights (temperature, then top-k and
    /// top-p), and the sum of the weights.
    fn distribution(&self, logits: &[f32]) -> (Vec<(u32, f32)>, f32) {
        let mut cand: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &v)| (i as u32, self.score(i, v))).collect();
        let k = if self.top_k == 0 { cand.len() } else { self.top_k.min(cand.len()) };
        if k < cand.len() {
            cand.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
            cand.truncate(k);
        }
        cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let max = cand[0].1;
        let mut total = 0.0;
        for c in cand.iter_mut() {
            c.1 = ((c.1 - max) / self.temperature).exp();
            total += c.1;
        }
        let mut cum = 0.0;
        let mut cut = cand.len();
        for (i, c) in cand.iter_mut().enumerate() {
            c.1 /= total;
            cum += c.1;
            if cum >= self.top_p {
                cut = i + 1;
                break;
            }
        }
        cand.truncate(cut);
        let mass: f32 = cand.iter().map(|c| c.1).sum();
        (cand, mass)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A word already said loses to one not said yet when they were close, and every word picked joins them;
    /// tokens that are not words are never held back.
    #[test]
    fn presence_penalty_steers_away_from_said_words() {
        let logits = [3.0f32, 2.6, 2.5, 0.0];
        let word = |t: u32| t != 3;
        let mut s = Sampler::new(0.0, 0, 1.0, 1).with_presence(1.0, [0u32, 3], word);
        assert!(!s.is_greedy(), "a penalty needs the host's pick, not the GPU's argmax");
        assert_eq!(s.sample(&logits), 1); // 0 was said: 2.0 < 2.6
        assert_eq!(s.sample(&logits), 2); // and now 1 too: 1.6 < 2.5
        assert_eq!(s.accept(&logits, 1), Err(0)); // all three said now: held back alike, 0 leads again
        assert_eq!(s.sample(&[0.0, 0.0, 0.0, 0.5]), 3); // not a word: never held back
        let mut plain = Sampler::new(0.0, 0, 1.0, 1);
        assert_eq!(plain.sample(&logits), 0);
        assert!(plain.is_greedy());
    }

    /// DRY: once the history has "10 11 12 13 14", saying "10 11 12" again makes 13 lose to a close second; one
    /// token said again lets anything follow (a run under DRY_ALLOWED), and a line break (a breaker) ends a run.
    #[test]
    fn dry_holds_back_rebuilding_an_earlier_run() {
        let logits = |favourite: u32| {
            let mut l = [0.0f32; 20];
            l[favourite as usize] = 2.0;
            l[19] = 1.5; // a close second
            l
        };
        let breaker = |t: u32| t == 9; // 9: a line break
        let mut s = Sampler::new(0.0, 0, 1.0, 1).with_dry(0.8, vec![10, 11, 12, 13, 14, 9, 10, 11, 12], breaker);
        assert!(!s.is_greedy());
        assert_eq!(s.sample(&logits(13)), 19); // a run of 3 held back: 0.8 x 1.75 = 1.4, 2.0 - 1.4 < 1.5
        let mut s = Sampler::new(0.0, 0, 1.0, 1).with_dry(0.8, vec![10, 12, 9, 10], breaker);
        assert_eq!(s.sample(&logits(12)), 12); // only "10" repeats: nothing held back
        let mut s = Sampler::new(0.0, 0, 1.0, 1).with_dry(0.8, vec![10, 11, 12, 9, 11], breaker);
        assert_eq!(s.sample(&logits(12)), 12); // "11" after a line break: the run is 1, not "9 11"
    }

    /// Drafting a fixed token and taking `accept`'s answer picks tokens with the same frequencies as `sample`.
    #[test]
    fn accepting_a_draft_keeps_the_distribution() {
        let logits = [2.0f32, 1.5, 1.0, 0.2, -1.0, -3.0];
        let runs = 200_000;
        let freq = |pick: &mut dyn FnMut(&mut Sampler) -> u32| {
            let mut s = Sampler::new(0.8, 5, 0.95, 7);
            let mut n = [0usize; 6];
            for _ in 0..runs {
                n[pick(&mut s) as usize] += 1;
            }
            n.map(|c| c as f64 / runs as f64)
        };
        let want = freq(&mut |s| s.sample(&logits));
        // a likely draft, an unlikely one, and one top-k or top-p leaves out
        for draft in [0u32, 3, 5] {
            let got = freq(&mut |s| match s.accept(&logits, draft) {
                Ok(()) => draft,
                Err(t) => t,
            });
            for (w, g) in want.iter().zip(&got) {
                assert!((w - g).abs() < 0.006, "draft {draft}: {want:?} vs {got:?}");
            }
        }
        assert_eq!(want[5], 0.0);
    }

    #[test]
    fn greedy_accepts_only_the_best() {
        let mut s = Sampler::new(0.0, 40, 0.9, 1);
        assert_eq!(s.accept(&[0.1, 3.0, 2.0], 1), Ok(()));
        assert_eq!(s.accept(&[0.1, 3.0, 2.0], 2), Err(1));
    }
}
