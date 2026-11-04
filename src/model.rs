//! Llama-family transformer on the CPU: Llama 2/3, SmolLM, Mistral-style GQA, Qwen3
//! (per-head RMSNorm on queries and keys, NEOX-style RoPE), DeepSeek-style models such as
//! GLM-4.7-Flash (multi-head latent attention, sigmoid-routed experts beside a shared one), and
//! Qwen3-Next (Gated DeltaNet layers, a recurrent kind of attention, between gated attention ones).

use rayon::prelude::*;

use crate::gguf::{Gguf, TensorInfo, Value};
use crate::profile;
use crate::quant::{dot_f32, gate_up_silu, matmul_many, Matrix, QuantizedBatch};

/// Prompts are processed this many tokens at a time (bounds scratch memory).
const MAX_BATCH: usize = 256;

#[derive(Debug, Clone)]
pub struct Config {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab: usize,
    pub max_ctx: usize,
    pub rope_dim: usize,
    pub rope_base: f32,
    /// NEOX RoPE rotates (x[i], x[i + rope_dim/2]); otherwise adjacent pairs (x[2i], x[2i+1]).
    /// GGUF converters permute llama's Q/K for adjacent pairs but leave Qwen's as trained.
    pub rope_neox: bool,
    /// Llama 3 frequency scaling (the rope_freqs tensor): per rotated pair, what its frequency is divided by
    pub rope_freqs: Option<Vec<f32>>,
    pub norm_eps: f32,
    /// Mixture of experts (qwen3moe; 0 for dense models): experts per layer, experts each token
    /// uses, and an expert's hidden size.
    pub n_experts: usize,
    pub n_experts_used: usize,
    pub expert_ffn_dim: usize,
    /// Multi-head latent attention (deepseek2); None for plain grouped-query attention.
    pub mla: Option<Mla>,
    /// deepseek2: the first layers keep a dense feed-forward; the others add always-on shared experts
    /// to the routed ones.
    pub n_dense_lead: usize,
    pub n_shared_experts: usize,
    /// deepseek2 routing: sigmoid scores (else softmax) picked with a per-expert bias, renormalised over
    /// the picks if `expert_weights_norm`, then times `expert_weights_scale`.
    pub gating_sigmoid: bool,
    pub expert_weights_norm: bool,
    pub expert_weights_scale: f32,
    /// qwen3next: Gated DeltaNet in every layer but each `every`-th; None: attention in every layer.
    pub delta: Option<Delta>,
    /// qwen3next: every query head's projection has a gate beside it ([q | gate] per head), and each head's
    /// attention output is scaled by sigmoid(gate).
    pub attn_gated: bool,
}

/// Gated DeltaNet sizes (qwen3next, qwen35moe). Each layer keeps a `head` x `head` state per value head
/// instead of a growing cache: every token decays it, corrects what it recalls for the token's key towards the
/// token's value, and reads it with the token's query. Value heads share key heads, `v_heads / k_heads` each.
#[derive(Debug, Clone, Copy)]
pub struct Delta {
    pub k_heads: usize,
    pub v_heads: usize,
    /// key and value width per head
    pub head: usize,
    /// causal convolution taps over q, k and v
    pub conv: usize,
    /// full attention every this many layers
    pub every: usize,
    /// qwen35moe: the value heads take the key heads in turn (llama.cpp's converter reorders them so), and
    /// beta and alpha come from two tensors in value head order; qwen3next: value heads grouped by key head,
    /// and one ssm_ba grouped the same way
    pub tiled: bool,
}

impl Delta {
    /// The key (and query) head value head `h` reads.
    pub fn key_head(&self, h: usize) -> usize {
        if self.tiled {
            h % self.k_heads
        } else {
            h / (self.v_heads / self.k_heads)
        }
    }

    /// Where value head `h`'s beta and alpha are among a batch's projected values (`n` tokens, token `b`):
    /// from two tensors, all the batch's betas and then all its alphas; from ssm_ba, per token and key head,
    /// its value heads' betas and then their alphas.
    pub fn ba_at(&self, n: usize, b: usize, h: usize) -> (usize, usize) {
        let v = self.v_heads;
        if self.tiled {
            return (b * v + h, (n + b) * v + h);
        }
        let per = v / self.k_heads;
        let beta = b * 2 * v + h / per * 2 * per + h % per;
        (beta, beta + per)
    }

    pub fn key_dim(&self) -> usize {
        self.k_heads * self.head
    }

    pub fn value_dim(&self) -> usize {
        self.v_heads * self.head
    }

    /// channels of the convolution: q, k and v side by side
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    /// floats of one layer's recurrent state
    pub fn state_len(&self) -> usize {
        self.v_heads * self.head * self.head
    }

    /// floats of one layer's convolution state (the last `conv - 1` inputs of every channel)
    pub fn conv_len(&self) -> usize {
        self.conv_dim() * (self.conv - 1)
    }
}

/// Multi-head latent attention sizes. Every head's keys and values come from one compressed vector per
/// token (`kv_lora` wide) plus a shared `rope_dim`-wide RoPE key, and only those are cached. Each head's
/// position-free query part (`nope` wide) is moved into the compressed space (attn_k_b), attention runs
/// over the cache as is, and each head's result is moved back out to `v_head` values (attn_v_b).
#[derive(Debug, Clone, Copy)]
pub struct Mla {
    /// width of the low-rank query step (0: queries come straight from attn_q)
    pub q_lora: usize,
    pub kv_lora: usize,
    pub nope: usize,
    pub v_head: usize,
}

pub const SUPPORTED_ARCHS: [&str; 6] = ["llama", "qwen3", "qwen3moe", "deepseek2", "qwen3next", "qwen35moe"];

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Config, String> {
        let arch = g.get("general.architecture").and_then(Value::as_str).unwrap_or("").to_string();
        if !SUPPORTED_ARCHS.contains(&arch.as_str()) {
            return Err(format!("architecture \"{arch}\" is not supported yet (supported: {})", SUPPORTED_ARCHS.join(", ")));
        }
        let u = |k: &str| g.get(&format!("{arch}.{k}")).and_then(Value::as_u64).map(|v| v as usize);
        let f = |k: &str| g.get(&format!("{arch}.{k}")).and_then(Value::as_f32);
        let need = |k: &str| u(k).ok_or_else(|| format!("missing {arch}.{k}"));

        let dim = need("embedding_length")?;
        let n_heads = need("attention.head_count")?;
        let deepseek = arch == "deepseek2";
        let mla = if deepseek {
            if g.tensor("blk.0.attn_k_b.weight").is_none() {
                return Err("this deepseek2 GGUF keeps attn_kv_b whole; convert it with a newer llama.cpp (attn_k_b / attn_v_b)".into());
            }
            if g.get(&format!("{arch}.rope.scaling.type")).and_then(Value::as_str).is_some_and(|t| t != "none") {
                return Err("RoPE scaling (YaRN) is not supported yet".into());
            }
            let rope = need("rope.dimension_count")?;
            // per-head sizes; in these GGUFs key_length / value_length describe the cache row instead
            let key = u("attention.key_length_mla").ok_or("missing attention.key_length_mla")?;
            let v_head = u("attention.value_length_mla").ok_or("missing attention.value_length_mla")?;
            Some(Mla { q_lora: u("attention.q_lora_rank").unwrap_or(0), kv_lora: need("attention.kv_lora_rank")?, nope: key - rope, v_head })
        } else {
            None
        };
        let head_dim = match mla {
            Some(m) => m.nope + need("rope.dimension_count")?,
            None => u("attention.key_length").unwrap_or(dim / n_heads),
        };
        let vocab = u("vocab_size").or_else(|| g.tensor("token_embd.weight").and_then(|t| t.dims.get(1)).map(|&v| v as usize)).ok_or("cannot tell the vocabulary size")?;
        let blocks = need("block_count")?;
        // Qwen3.5 / 3.6 are Qwen3-Next's design with the value heads reordered (see Delta::tiled); for text,
        // their multi-section RoPE is Qwen3-Next's partial RoPE (every section reads the same position)
        let next = arch == "qwen3next" || arch == "qwen35moe";
        let delta = if next {
            let d = Delta {
                k_heads: need("ssm.group_count")?,
                v_heads: need("ssm.time_step_rank")?,
                head: need("ssm.state_size")?,
                conv: need("ssm.conv_kernel")?,
                every: u("full_attention_interval").unwrap_or(4),
                tiled: arch == "qwen35moe",
            };
            if need("ssm.inner_size")? != d.value_dim() || d.v_heads % d.k_heads != 0 || d.conv < 2 {
                return Err(format!("this {arch} file's DeltaNet sizes do not fit together"));
            }
            Some(d)
        } else {
            None
        };
        let cfg = Config {
            dim,
            // multi-token-prediction blocks come after the real ones; they are not run
            n_layers: blocks - u("nextn_predict_layers").unwrap_or(0),
            n_heads,
            // latent attention: one shared key-value "head" for all query heads
            n_kv_heads: if mla.is_some() { 1 } else { u("attention.head_count_kv").unwrap_or(n_heads) },
            head_dim,
            // Qwen3.5's files have no dense feed-forward size; its shared expert's is the widest there is
            ffn_dim: u("feed_forward_length").or_else(|| u("expert_shared_feed_forward_length")).ok_or(format!("missing {arch}.feed_forward_length"))?,
            // not every GGUF has <arch>.vocab_size; the embedding table always says
            vocab,
            max_ctx: need("context_length")?,
            rope_dim: u("rope.dimension_count").unwrap_or(head_dim),
            rope_base: f("rope.freq_base").unwrap_or(10_000.0),
            rope_neox: arch.starts_with("qwen3"),
            rope_freqs: match g.tensor("rope_freqs.weight") {
                Some(_) => Some(vector(g, "rope_freqs.weight")?),
                None => None,
            },
            norm_eps: f("attention.layer_norm_rms_epsilon").unwrap_or(1e-5),
            n_experts: u("expert_count").unwrap_or(0),
            n_experts_used: u("expert_used_count").unwrap_or(0),
            expert_ffn_dim: u("expert_feed_forward_length").unwrap_or(0),
            mla,
            n_dense_lead: u("leading_dense_block_count").unwrap_or(0),
            n_shared_experts: u("expert_shared_count").unwrap_or(0),
            // 1 softmax, 2 sigmoid; older GLM-4.7-Flash files lack the key, and llama.cpp then picks sigmoid
            gating_sigmoid: match u("expert_gating_func") {
                Some(f) => f == 2,
                None => deepseek && vocab == 154_880 && matches!(blocks, 47 | 48),
            },
            expert_weights_norm: g.get(&format!("{arch}.expert_weights_norm")).and_then(Value::as_bool).unwrap_or(false),
            expert_weights_scale: f("expert_weights_scale").unwrap_or(1.0),
            delta,
            attn_gated: next,
        };
        if cfg.rope_freqs.as_ref().is_some_and(|f| f.len() != cfg.rope_dim / 2) {
            return Err(format!("rope_freqs holds {} values, not one per rotated pair ({})", cfg.rope_freqs.as_ref().unwrap().len(), cfg.rope_dim / 2));
        }
        Ok(cfg)
    }

    pub fn is_moe(&self) -> bool {
        self.n_experts > 0
    }

    /// Layer `l` is a Gated DeltaNet one (no attention, so no cache rows).
    pub fn recurrent(&self, l: usize) -> bool {
        self.delta.is_some_and(|d| (l + 1) % d.every != 0)
    }

    /// Layers with attention, which keep cache rows.
    pub fn n_attn_layers(&self) -> usize {
        (0..self.n_layers).filter(|&l| !self.recurrent(l)).count()
    }

    /// Where layer `l` keeps its rows: its place among the attention layers (or among the DeltaNet ones,
    /// for its states).
    pub fn slot(&self, l: usize) -> usize {
        (0..l).filter(|&i| self.recurrent(i) == self.recurrent(l)).count()
    }

    fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// Floats cached per token and layer for keys and for values. Latent attention keeps the compressed
    /// vector with the RoPE key after it, and the values are that same vector.
    pub fn k_row(&self) -> usize {
        self.mla.map_or(self.kv_dim(), |m| m.kv_lora + self.rope_dim)
    }

    pub fn v_row(&self) -> usize {
        self.mla.map_or(self.kv_dim(), |m| m.kv_lora)
    }

    /// Attention as the kernel sees it: key and value width per kv head, and the score scale.
    fn att_dims(&self) -> (usize, usize, f32) {
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        match self.mla {
            Some(m) => (m.kv_lora + self.rope_dim, m.kv_lora, scale),
            None => (self.head_dim, self.head_dim, scale),
        }
    }

    fn q_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
}

struct Layer<'a> {
    attn_norm: Vec<f32>,
    attn: Attn<'a>,
    wo: Matrix<'a>, // attn_output, or a DeltaNet layer's ssm_out
    ffn_norm: Vec<f32>,
    ffn: Ffn<'a>,
}

/// A Gated DeltaNet layer's weights (see `Delta`).
pub struct DeltaLayer<'a> {
    /// input to q, k and v side by side (each head's q and k are shared by `v_heads / k_heads` value heads)
    pub wqkv: Matrix<'a>,
    /// input to z, the gate of the output norm
    pub wz: Matrix<'a>,
    /// input to beta (how far a token corrects the state) and alpha (how fast the state decays): ssm_ba
    /// grouped by key head (its value heads' betas, then their alphas), or ssm_beta and ssm_alpha (tiled)
    pub wba: Vec<Matrix<'a>>,
    /// per channel of q, k, v: the convolution taps, oldest input first
    pub conv: Vec<f32>,
    /// per value head: added to alpha before softplus
    pub dt_bias: Vec<f32>,
    /// per value head: -exp(A_log), the decay rate's scale
    pub a: Vec<f32>,
    /// RMSNorm weights over each value head's output (used as is, without the +1 of the other norms)
    pub norm: Vec<f32>,
}

impl<'a> DeltaLayer<'a> {
    fn load(g: &'a Gguf, l: usize, d: &Delta) -> Result<DeltaLayer<'a>, String> {
        let p = |s: &str| format!("blk.{l}.{s}");
        let layer = DeltaLayer {
            wqkv: matrix(g, &p("attn_qkv.weight"))?,
            wz: matrix(g, &p("attn_gate.weight"))?,
            wba: if d.tiled {
                vec![matrix(g, &p("ssm_beta.weight"))?, matrix(g, &p("ssm_alpha.weight"))?]
            } else {
                vec![matrix(g, &p("ssm_ba.weight"))?]
            },
            conv: vector(g, &p("ssm_conv1d.weight"))?,
            dt_bias: vector(g, &p("ssm_dt.bias"))?,
            a: vector(g, &p("ssm_a"))?,
            norm: vector(g, &p("ssm_norm.weight"))?,
        };
        let ba_rows: usize = layer.wba.iter().map(|m| m.rows).sum();
        if layer.wqkv.rows != d.conv_dim() || layer.wz.rows != d.value_dim() || ba_rows != 2 * d.v_heads || layer.conv.len() != d.conv_dim() * d.conv || layer.a.len() != d.v_heads || layer.norm.len() != d.head {
            return Err(format!("layer {l}: DeltaNet tensor shapes do not match the model's sizes (a file with ssm_in instead of attn_qkv needs a newer conversion)"));
        }
        Ok(layer)
    }
}

enum Attn<'a> {
    Gqa {
        wq: Matrix<'a>,
        wk: Matrix<'a>,
        wv: Matrix<'a>,
        q_norm: Option<Vec<f32>>, // Qwen3: RMSNorm over each query head before RoPE
        k_norm: Option<Vec<f32>>,
    },
    /// See `Mla`. `wq_a` and its norm are the low-rank query step (absent: `wq` reads the input directly).
    Mla {
        wq_a: Option<(Matrix<'a>, Vec<f32>)>,
        wq: Matrix<'a>,
        wkv_a: Matrix<'a>,
        kv_a_norm: Vec<f32>,
        k_b: Vec<Matrix<'a>>, // per head: kv_lora rows of nope
        v_b: Vec<Matrix<'a>>, // per head: v_head rows of kv_lora
    },
    Delta(Box<DeltaLayer<'a>>),
}

enum Ffn<'a> {
    Dense { gate: Matrix<'a>, up: Matrix<'a>, down: Matrix<'a> },
    /// router: n_experts rows of dim f32 logit weights; bias: deepseek2's per-expert pick bias;
    /// shared: the always-on expert(s), as one SwiGLU block; shared_gate: qwen3next scales its output by
    /// sigmoid(shared_gate · x)
    Moe { router: Vec<f32>, bias: Option<Vec<f32>>, experts: Vec<Expert<'a>>, shared: Option<Expert<'a>>, shared_gate: Option<Vec<f32>> },
}

/// One expert's SwiGLU feed-forward, inside the model file.
#[derive(Clone, Copy)]
pub struct Expert<'a> {
    pub gate: Matrix<'a>,
    pub up: Matrix<'a>,
    pub down: Matrix<'a>,
}

pub struct Model<'a> {
    pub cfg: Config,
    token_embd: Matrix<'a>,
    layers: Vec<Layer<'a>>,
    output_norm: Vec<f32>,
    output: Matrix<'a>,
}

fn tensor<'g>(g: &'g Gguf, name: &str) -> Result<&'g TensorInfo, String> {
    g.tensor(name).ok_or_else(|| format!("missing tensor {name}"))
}

fn matrix<'g>(g: &'g Gguf, name: &str) -> Result<Matrix<'g>, String> {
    let t = tensor(g, name)?;
    if t.ty.block().is_none() {
        return Err(format!("tensor {name} uses {} which is not supported yet", t.ty.name()));
    }
    if t.dims.len() != 2 {
        return Err(format!("tensor {name} is not 2-D"));
    }
    Ok(Matrix { data: &g.mmap[t.offset..t.offset + t.n_bytes], ty: t.ty, cols: t.dims[0] as usize, rows: t.dims[1] as usize })
}

/// A 3-D expert tensor (cols x rows x experts, every expert's matrix stored whole) as one matrix per expert.
fn expert_matrices<'g>(g: &'g Gguf, name: &str) -> Result<Vec<Matrix<'g>>, String> {
    let t = tensor(g, name)?;
    if t.ty.block().is_none() {
        return Err(format!("tensor {name} uses {} which is not supported yet", t.ty.name()));
    }
    let [cols, rows, n] = t.dims[..] else { return Err(format!("tensor {name} is not 3-D")) };
    let (cols, rows, n) = (cols as usize, rows as usize, n as usize);
    let bytes = t.n_bytes / n;
    Ok((0..n).map(|e| Matrix { data: &g.mmap[t.offset + e * bytes..t.offset + (e + 1) * bytes], ty: t.ty, rows, cols }).collect())
}

/// Every layer's experts (none for a dense model; empty for deepseek2's dense lead layers), for the CPU
/// expert path of the GPU backend too.
pub fn load_experts<'g>(g: &'g Gguf, cfg: &Config) -> Result<Vec<Vec<Expert<'g>>>, String> {
    if !cfg.is_moe() {
        return Ok(Vec::new());
    }
    (0..cfg.n_layers)
        .map(|i| {
            if i < cfg.n_dense_lead {
                return Ok(Vec::new()); // a dense layer
            }
            let p = |s: &str| format!("blk.{i}.{s}.weight");
            let (gate, up, down) = (expert_matrices(g, &p("ffn_gate_exps"))?, expert_matrices(g, &p("ffn_up_exps"))?, expert_matrices(g, &p("ffn_down_exps"))?);
            if gate.len() != cfg.n_experts || up.len() != cfg.n_experts || down.len() != cfg.n_experts {
                return Err(format!("layer {i}: expert tensors do not hold {} experts", cfg.n_experts));
            }
            if gate[0].rows != cfg.expert_ffn_dim || down[0].cols != cfg.expert_ffn_dim || gate[0].cols != cfg.dim {
                return Err(format!("layer {i}: expert shapes do not match expert_feed_forward_length {}", cfg.expert_ffn_dim));
            }
            Ok(gate.into_iter().zip(up).zip(down).map(|((gate, up), down)| Expert { gate, up, down }).collect())
        })
        .collect()
}

/// Qwen3-MoE routing for one token: the `k` largest router logits, weighted by a softmax over
/// just those k (the same as a softmax over all experts renormalised to the chosen ones).
pub fn route(logits: &[f32], k: usize, out: &mut Vec<(usize, f32)>) {
    out.clear();
    out.extend(logits.iter().copied().enumerate());
    // descending logit; ties to the lower expert index
    out.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out.truncate(k);
    out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let max = out[0].1;
    let total: f32 = out.iter().map(|&(_, l)| (l - max).exp()).sum();
    for e in out.iter_mut() {
        e.1 = (e.1 - max).exp() / total;
    }
}

/// deepseek2 routing for one token: scores are sigmoid(logit) (or a softmax over every expert), the `k`
/// best by score + `bias` are picked, and weighted by their scores, renormalised to sum to 1 if `norm`,
/// then times `scale`.
pub fn route_deepseek(logits: &[f32], bias: Option<&[f32]>, k: usize, sigmoid: bool, norm: bool, scale: f32, out: &mut Vec<(usize, f32)>) {
    let mut scores: Vec<f32> = if sigmoid {
        logits.iter().map(|&l| 1.0 / (1.0 + (-l).exp())).collect()
    } else {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
        let total: f32 = e.iter().sum();
        e.into_iter().map(|v| v / total).collect()
    };
    out.clear();
    out.extend(scores.iter().enumerate().map(|(e, &sc)| (e, sc + bias.map_or(0.0, |b| b[e]))));
    out.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out.truncate(k);
    out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let total: f32 = out.iter().map(|&(e, _)| scores[e]).sum();
    let div = if norm { total.max(f32::MIN_POSITIVE) } else { 1.0 };
    for p in out.iter_mut() {
        p.1 = scores[p.0] / div * scale;
    }
    scores.clear();
}

/// deepseek2's `moe_route`: every row's router logits, then `route_deepseek` with the model's settings.
pub fn moe_route_deepseek(c: &Config, router: &[f32], bias: Option<&[f32]>, xb: &[f32], picks: &mut Vec<(usize, f32)>, sc: &mut MoeScratch) {
    picks.clear();
    let mut top = Vec::new();
    for row in xb.chunks_exact(c.dim) {
        sc.logits.clear();
        sc.logits.extend(router.chunks_exact(c.dim).map(|w| dot_f32(w, row)));
        route_deepseek(&sc.logits, bias, c.n_experts_used, c.gating_sigmoid, c.expert_weights_norm, c.expert_weights_scale, &mut top);
        picks.extend_from_slice(&top);
    }
}

fn vector(g: &Gguf, name: &str) -> Result<Vec<f32>, String> {
    let t = tensor(g, name)?;
    if t.ty.block().is_none() {
        return Err(format!("tensor {name} uses {} which is not supported yet", t.ty.name()));
    }
    let n = t.n_elements() as usize;
    let mut out = vec![0f32; n];
    let m = Matrix { data: &g.mmap[t.offset..t.offset + t.n_bytes], ty: t.ty, rows: 1, cols: n };
    m.row_f32(0, &mut out);
    Ok(out)
}

impl<'a> Model<'a> {
    pub fn load(g: &'a Gguf) -> Result<Model<'a>, String> {
        let cfg = Config::from_gguf(g)?;
        let token_embd = matrix(g, "token_embd.weight")?;
        let mut experts = load_experts(g, &cfg)?.into_iter();
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}.weight");
            let ffn = match experts.next().filter(|e| !e.is_empty()) {
                Some(experts) => Ffn::Moe {
                    router: vector(g, &p("ffn_gate_inp"))?,
                    bias: g.tensor(&format!("blk.{i}.exp_probs_b.bias")).map(|_| vector(g, &format!("blk.{i}.exp_probs_b.bias"))).transpose()?,
                    experts,
                    shared: match g.tensor(&p("ffn_gate_shexp")) {
                        Some(_) => Some(Expert { gate: matrix(g, &p("ffn_gate_shexp"))?, up: matrix(g, &p("ffn_up_shexp"))?, down: matrix(g, &p("ffn_down_shexp"))? }),
                        None => None,
                    },
                    shared_gate: g.tensor(&p("ffn_gate_inp_shexp")).map(|_| vector(g, &p("ffn_gate_inp_shexp"))).transpose()?,
                },
                None => Ffn::Dense { gate: matrix(g, &p("ffn_gate"))?, up: matrix(g, &p("ffn_up"))?, down: matrix(g, &p("ffn_down"))? },
            };
            let attn = match cfg.mla {
                _ if cfg.recurrent(i) => Attn::Delta(Box::new(DeltaLayer::load(g, i, cfg.delta.as_ref().unwrap())?)),
                Some(m) => {
                    let (k_b, v_b) = (expert_matrices(g, &p("attn_k_b"))?, expert_matrices(g, &p("attn_v_b"))?);
                    if k_b.len() != cfg.n_heads || k_b[0].cols != m.nope || k_b[0].rows != m.kv_lora || v_b.len() != cfg.n_heads || v_b[0].cols != m.kv_lora || v_b[0].rows != m.v_head {
                        return Err(format!("layer {i}: attn_k_b / attn_v_b shapes do not match the attention sizes"));
                    }
                    Attn::Mla {
                        wq_a: if m.q_lora > 0 { Some((matrix(g, &p("attn_q_a"))?, vector(g, &p("attn_q_a_norm"))?)) } else { None },
                        wq: matrix(g, &p(if m.q_lora > 0 { "attn_q_b" } else { "attn_q" }))?,
                        wkv_a: matrix(g, &p("attn_kv_a_mqa"))?,
                        kv_a_norm: vector(g, &p("attn_kv_a_norm"))?,
                        k_b,
                        v_b,
                    }
                }
                None => Attn::Gqa {
                    wq: matrix(g, &p("attn_q"))?,
                    wk: matrix(g, &p("attn_k"))?,
                    wv: matrix(g, &p("attn_v"))?,
                    q_norm: g.tensor(&p("attn_q_norm")).map(|_| vector(g, &p("attn_q_norm"))).transpose()?,
                    k_norm: g.tensor(&p("attn_k_norm")).map(|_| vector(g, &p("attn_k_norm"))).transpose()?,
                },
            };
            let wo = matrix(g, &p(if cfg.recurrent(i) { "ssm_out" } else { "attn_output" }))?;
            // qwen3next names the norm before the feed-forward after the attention it follows
            let ffn_norm = if g.tensor(&p("ffn_norm")).is_some() { vector(g, &p("ffn_norm"))? } else { vector(g, &p("post_attention_norm"))? };
            layers.push(Layer { attn_norm: vector(g, &p("attn_norm"))?, attn, wo, ffn_norm, ffn });
        }
        // small models tie the output projection to the embedding table
        let output = if g.tensor("output.weight").is_some() { matrix(g, "output.weight")? } else { token_embd };
        Ok(Model { output_norm: vector(g, "output_norm.weight")?, cfg, token_embd, layers, output })
    }
}

/// Per-sequence state: KV cache plus scratch buffers that grow to the largest batch seen.
pub struct Session {
    pub pos: usize,
    pub ctx: usize,
    k_cache: Vec<f32>, // [attention layer][pos][k_row]
    v_cache: Vec<f32>, // [attention layer][pos][v_row]
    // Gated DeltaNet layers: [layer][channel][conv - 1] latest inputs (oldest first), and
    // [layer][value head][value][key] states; and copies of both at earlier positions (`snapshot`)
    conv_state: Vec<f32>,
    ssm_state: Vec<f32>,
    snaps: Vec<(usize, Vec<f32>, Vec<f32>)>,
    // DeltaNet scratch: q k v (then convolved), z, beta and alpha per token; gated attention: [q | gate] per
    // head as projected, then the gates
    dqkv: Vec<f32>,
    dz: Vec<f32>,
    dba: Vec<f32>,
    qg: Vec<f32>,
    att_gate: Vec<f32>,
    x: Vec<f32>,
    xb: Vec<f32>,
    xq: QuantizedBatch,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att_out: Vec<f32>,
    att_partial: Vec<f32>,
    proj: Vec<f32>,
    gate: Vec<f32>,
    picks: Vec<(usize, f32)>, // mixture of experts: (expert, weight) per token and pick
    moe: MoeScratch,
    // latent attention: the low-rank queries, the queries moved into the latent space, every head's
    // output moved back out, and one head's rows gathered for its own projection
    lat: Vec<f32>,
    qa: Vec<f32>,
    heads: Vec<f32>,
    head_in: Vec<f32>,
    head_out: Vec<f32>,
    pub logits: Vec<f32>,
    rope_inv_freq: Vec<f32>,
}

impl Session {
    pub fn new(cfg: &Config, ctx: usize) -> Session {
        let ctx = ctx.min(cfg.max_ctx);
        let half = cfg.rope_dim / 2;
        let deltas = cfg.n_layers - cfg.n_attn_layers();
        Session {
            pos: 0,
            ctx,
            k_cache: vec![0.0; cfg.n_attn_layers() * ctx * cfg.k_row()],
            v_cache: vec![0.0; cfg.n_attn_layers() * ctx * cfg.v_row()],
            conv_state: vec![0.0; cfg.delta.map_or(0, |d| deltas * d.conv_len())],
            ssm_state: vec![0.0; cfg.delta.map_or(0, |d| deltas * d.state_len())],
            snaps: Vec::new(),
            dqkv: Vec::new(),
            dz: Vec::new(),
            dba: Vec::new(),
            qg: Vec::new(),
            att_gate: Vec::new(),
            x: Vec::new(),
            xb: Vec::new(),
            xq: QuantizedBatch::default(),
            q: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            att_out: Vec::new(),
            att_partial: Vec::new(),
            proj: Vec::new(),
            gate: Vec::new(),
            picks: Vec::new(),
            moe: MoeScratch::default(),
            lat: Vec::new(),
            qa: Vec::new(),
            heads: Vec::new(),
            head_in: Vec::new(),
            head_out: Vec::new(),
            logits: vec![0.0; cfg.vocab],
            rope_inv_freq: (0..half).map(|i| cfg.rope_base.powf(-(2.0 * i as f32) / cfg.rope_dim as f32) / cfg.rope_freqs.as_ref().map_or(1.0, |f| f[i])).collect(),
        }
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        self.conv_state.fill(0.0);
        self.ssm_state.fill(0.0);
        self.snaps.clear();
    }

    /// The model keeps a recurrent state (Gated DeltaNet), which cannot be cut back to an earlier position
    /// like a cache; only a snapshot (or the start) can be gone back to.
    pub fn recurrent(&self) -> bool {
        !self.ssm_state.is_empty()
    }

    /// Keeps a copy of the recurrent state at the current position (the latest two are kept).
    pub fn snapshot(&mut self) {
        if !self.recurrent() || self.snaps.iter().any(|s| s.0 == self.pos) {
            return;
        }
        // three: the earliest stays (a prompt's shared opening), with the latest prompt's header and end
        if self.snaps.len() == 3 {
            self.snaps.remove(1);
        }
        self.snaps.push((self.pos, self.conv_state.clone(), self.ssm_state.clone()));
    }

    /// Goes back to at most `pos` tokens and returns how many are kept: all of them for a model without a
    /// recurrent state, else the latest snapshot at or before `pos` (or none).
    pub fn truncate(&mut self, pos: usize) -> usize {
        if pos >= self.pos {
            return self.pos;
        }
        if !self.recurrent() {
            self.pos = pos;
            return pos;
        }
        self.snaps.retain(|s| s.0 <= pos);
        match self.snaps.iter().max_by_key(|s| s.0) {
            Some((at, conv, ssm)) => {
                self.conv_state.copy_from_slice(conv);
                self.ssm_state.copy_from_slice(ssm);
                self.pos = *at;
            }
            None => {
                self.conv_state.fill(0.0);
                self.ssm_state.fill(0.0);
                self.pos = 0;
            }
        }
        self.pos
    }

    fn size_for(&mut self, cfg: &Config, n: usize) {
        let (kd, vd, _) = cfg.att_dims();
        let m = cfg.mla.unwrap_or(Mla { q_lora: 0, kv_lora: 0, nope: 0, v_head: 0 });
        let d = cfg.delta.unwrap_or(Delta { k_heads: 0, v_heads: 0, head: 0, conv: 0, every: 1, tiled: false });
        for (buf, width) in [
            (&mut self.x, cfg.dim),
            (&mut self.xb, cfg.dim),
            (&mut self.q, cfg.q_dim()),
            (&mut self.k, cfg.k_row()),
            (&mut self.v, cfg.kv_dim()),
            (&mut self.att_out, (cfg.n_heads * vd).max(d.value_dim())),
            (&mut self.dqkv, d.conv_dim()),
            (&mut self.dz, d.value_dim()),
            (&mut self.dba, 2 * d.v_heads),
            (&mut self.qg, if cfg.attn_gated { 2 * cfg.q_dim() } else { 0 }),
            (&mut self.att_gate, if cfg.attn_gated { cfg.q_dim() } else { 0 }),
            (&mut self.proj, cfg.dim),
            (&mut self.gate, cfg.ffn_dim.max(cfg.expert_ffn_dim * cfg.n_shared_experts)),
            (&mut self.lat, m.q_lora),
            (&mut self.qa, if cfg.mla.is_some() { cfg.n_heads * kd } else { 0 }),
            (&mut self.heads, cfg.n_heads * m.v_head),
        ] {
            buf.resize(n * width, 0.0);
        }
    }
}

/// Scratch for the CPU expert path, reused between calls.
#[derive(Default)]
pub struct MoeScratch {
    one: crate::quant::ExpertScratch,
    logits: Vec<f32>,
    by_expert: Vec<Vec<(usize, f32)>>, // per expert: (token, weight)
    xin: Vec<f32>,
    xq: QuantizedBatch,
    h: Vec<f32>,
    out: Vec<f32>,
}

/// Routes each of the `xb.len() / dim` (normalised) rows: `picks` gets k (expert, weight) pairs per row.
pub fn moe_route(router: &[f32], k: usize, xb: &[f32], dim: usize, picks: &mut Vec<(usize, f32)>, sc: &mut MoeScratch) {
    let n_experts = router.len() / dim;
    picks.clear();
    let mut top = Vec::with_capacity(n_experts);
    for row in xb.chunks_exact(dim) {
        sc.logits.clear();
        sc.logits.extend(router.chunks_exact(dim).map(|w| dot_f32(w, row)));
        route(&sc.logits, k, &mut top);
        picks.extend_from_slice(&top);
    }
}

/// `acc[b] += Σ weight · expert(xb[b])` over the k experts `picks` gives row b (`expert(e)` hands out
/// expert e of `n_experts`). Rows sharing an expert run through it together, so each expert's weights
/// are read once per batch.
#[allow(clippy::too_many_arguments)]
pub fn moe_experts<'e>(n_experts: usize, expert: impl Fn(usize) -> Expert<'e>, picks: &[(usize, f32)], k: usize, xb: &[f32], dim: usize, acc: &mut [f32], sc: &mut MoeScratch) {
    // one token: all its experts at once on the spinning pool (row blocks of 32 for the quantised hidden)
    if xb.len() == dim && !picks.is_empty() && !crate::quant::exact_mode() {
        let ex: Vec<_> = picks
            .iter()
            .map(|&(e, w)| {
                let x = expert(e);
                (x.gate, x.up, x.down, w)
            })
            .collect();
        if ex[0].0.rows % 32 == 0 {
            return crate::quant::experts_one(&ex, xb, acc, &mut sc.one);
        }
    }
    sc.by_expert.resize_with(n_experts, Vec::new);
    sc.by_expert.iter_mut().for_each(Vec::clear);
    for (i, &(e, w)) in picks.iter().enumerate() {
        sc.by_expert[e].push((i / k, w));
    }
    for (e, rows) in sc.by_expert.iter().enumerate().filter(|(_, r)| !r.is_empty()) {
        let ex = &expert(e);
        let (m, hidden) = (rows.len(), ex.gate.rows);
        sc.xin.clear();
        for &(b, _) in rows {
            sc.xin.extend_from_slice(&xb[b * dim..(b + 1) * dim]);
        }
        sc.h.resize(m * hidden, 0.0);
        sc.out.resize(m * dim, 0.0);
        sc.xq.quantize(&sc.xin, dim);
        gate_up_silu(&ex.gate, &ex.up, &sc.xin, &sc.xq, &mut sc.h);
        sc.xq.quantize(&sc.h, hidden);
        ex.down.matmul(&sc.h, &sc.xq, &mut sc.out);
        for (&(b, w), o) in rows.iter().zip(sc.out.chunks_exact(dim)) {
            for (a, v) in acc[b * dim..(b + 1) * dim].iter_mut().zip(o) {
                *a += w * v;
            }
        }
    }
}

fn rms_norm_rows(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    let dim = w.len();
    for (o, xr) in out.chunks_exact_mut(dim).zip(x.chunks_exact(dim)) {
        let ms = dot_f32(xr, xr) / dim as f32;
        let scale = 1.0 / (ms + eps).sqrt();
        for ((ov, v), g) in o.iter_mut().zip(xr).zip(w) {
            *ov = v * scale * g;
        }
    }
}

/// RMSNorm of every `w.len()`-sized chunk in place (Qwen3's per-head query/key norm).
fn rms_norm_inplace(buf: &mut [f32], w: &[f32], eps: f32) {
    for chunk in buf.chunks_exact_mut(w.len()) {
        let scale = 1.0 / (dot_f32(chunk, chunk) / w.len() as f32 + eps).sqrt();
        for (v, g) in chunk.iter_mut().zip(w) {
            *v *= scale * g;
        }
    }
}

/// Rotary position embedding on every head of `row`. Pairs are (x[2i], x[2i+1]) for GGUF llama
/// weights, or (x[i], x[i + rope_dim/2]) for NEOX-style models such as Qwen.
fn rope(row: &mut [f32], head_dim: usize, rope_dim: usize, pos: usize, inv_freq: &[f32], neox: bool) {
    let half = rope_dim / 2;
    for head in row.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let (s, c) = (pos as f32 * inv_freq[i]).sin_cos();
            let (ia, ib) = if neox { (i, i + half) } else { (2 * i, 2 * i + 1) };
            let (a, b) = (head[ia], head[ib]);
            head[ia] = a * c - b * s;
            head[ib] = a * s + b * c;
        }
    }
}

impl Model<'_> {
    /// Feeds a whole prompt; leaves the logits for the token after it in `s.logits`.
    pub fn prefill(&self, tokens: &[u32], s: &mut Session) -> Result<(), String> {
        let chunks: Vec<&[u32]> = tokens.chunks(MAX_BATCH).collect();
        for (i, chunk) in chunks.iter().enumerate() {
            self.forward_batch(chunk, s, i + 1 == chunks.len())?;
        }
        Ok(())
    }

    /// Runs one token; leaves next-token logits in `s.logits` if `want_logits`.
    pub fn forward(&self, token: u32, s: &mut Session, want_logits: bool) -> Result<(), String> {
        self.forward_batch(&[token], s, want_logits)
    }

    /// Runs `tokens` at positions `s.pos..s.pos + n` in one pass, so every weight is read once per batch.
    fn forward_batch(&self, tokens: &[u32], s: &mut Session, want_logits: bool) -> Result<(), String> {
        let c = &self.cfg;
        let n = tokens.len();
        if n == 0 {
            return Ok(());
        }
        if s.pos + n > s.ctx {
            return Err(format!("context full ({} tokens)", s.ctx));
        }
        if let Some(&bad) = tokens.iter().find(|&&t| t as usize >= self.token_embd.rows) {
            return Err(format!("token id {bad} out of range"));
        }
        s.size_for(c, n);
        let (dim, kv_dim, q_dim, hd) = (c.dim, c.kv_dim(), c.q_dim(), c.head_dim);
        let pos0 = s.pos;

        profile::time(0, || {
            for (b, &t) in tokens.iter().enumerate() {
                self.token_embd.row_f32(t as usize, &mut s.x[b * dim..(b + 1) * dim]);
            }
        });

        for (l, layer) in self.layers.iter().enumerate() {
            // --- attention ---
            profile::time(1, || {
                rms_norm_rows(&mut s.xb, &s.x, &layer.attn_norm, c.norm_eps);
                s.xq.quantize(&s.xb, dim);
            });
            let (k_base, v_base) = (c.slot(l) * s.ctx * c.k_row(), c.slot(l) * s.ctx * c.v_row());
            match &layer.attn {
                Attn::Gqa { wq, wk, wv, q_norm, k_norm } => {
                    profile::time(2, || {
                        if c.attn_gated {
                            // [q | gate] per head: the queries to s.q, the gates aside
                            matmul_many(&mut [(wq, &mut s.qg), (wk, &mut s.k), (wv, &mut s.v)], &s.xb, &s.xq);
                            for (pair, (q, g)) in s.qg.chunks_exact(2 * hd).zip(s.q.chunks_exact_mut(hd).zip(s.att_gate.chunks_exact_mut(hd))) {
                                q.copy_from_slice(&pair[..hd]);
                                g.copy_from_slice(&pair[hd..]);
                            }
                        } else {
                            matmul_many(&mut [(wq, &mut s.q), (wk, &mut s.k), (wv, &mut s.v)], &s.xb, &s.xq);
                        }
                    });
                    profile::time(3, || {
                        if let Some(w) = q_norm {
                            rms_norm_inplace(&mut s.q, w, c.norm_eps);
                        }
                        if let Some(w) = k_norm {
                            rms_norm_inplace(&mut s.k, w, c.norm_eps);
                        }
                        for b in 0..n {
                            rope(&mut s.q[b * q_dim..(b + 1) * q_dim], hd, c.rope_dim, pos0 + b, &s.rope_inv_freq, c.rope_neox);
                            rope(&mut s.k[b * kv_dim..(b + 1) * kv_dim], hd, c.rope_dim, pos0 + b, &s.rope_inv_freq, c.rope_neox);
                        }
                        let at = k_base + pos0 * kv_dim;
                        s.k_cache[at..at + n * kv_dim].copy_from_slice(&s.k);
                        s.v_cache[at..at + n * kv_dim].copy_from_slice(&s.v);
                    });
                }
                Attn::Mla { wq_a, wq, wkv_a, kv_a_norm, k_b, .. } => profile::time(2, || self.mla_in(s, n, k_base, v_base, wq_a.as_ref(), wq, wkv_a, kv_a_norm, k_b)),
                Attn::Delta(d) => profile::time(2, || self.delta_net(d, s, l, n)),
            }
            if !matches!(layer.attn, Attn::Delta(_)) {
                profile::time(4, || self.attention(s, k_base, v_base, n));
            }
            if c.attn_gated && !matches!(layer.attn, Attn::Delta(_)) {
                for (o, g) in s.att_out[..n * q_dim].iter_mut().zip(&s.att_gate) {
                    *o *= 1.0 / (1.0 + (-g).exp());
                }
            }

            profile::time(5, || {
                let heads = match &layer.attn {
                    Attn::Gqa { .. } => &s.att_out[..n * q_dim],
                    Attn::Delta(_) => &s.att_out[..n * layer.wo.cols],
                    Attn::Mla { v_b, .. } => {
                        // every head's latent result out to its values
                        let (m, vd) = (c.mla.unwrap(), c.att_dims().1);
                        for (h, vb) in v_b.iter().enumerate() {
                            s.head_in.clear();
                            for b in 0..n {
                                let o = (b * c.n_heads + h) * vd;
                                s.head_in.extend_from_slice(&s.att_out[o..o + vd]);
                            }
                            s.xq.quantize(&s.head_in, vd);
                            s.head_out.resize(n * m.v_head, 0.0);
                            vb.matmul(&s.head_in, &s.xq, &mut s.head_out);
                            for b in 0..n {
                                let o = (b * c.n_heads + h) * m.v_head;
                                s.heads[o..o + m.v_head].copy_from_slice(&s.head_out[b * m.v_head..(b + 1) * m.v_head]);
                            }
                        }
                        &s.heads
                    }
                };
                s.xq.quantize(heads, layer.wo.cols);
                layer.wo.matmul(heads, &s.xq, &mut s.proj);
                for (x, p) in s.x.iter_mut().zip(&s.proj) {
                    *x += p;
                }
            });

            // --- feed-forward (SwiGLU) ---
            profile::time(1, || {
                rms_norm_rows(&mut s.xb, &s.x, &layer.ffn_norm, c.norm_eps);
                s.xq.quantize(&s.xb, dim);
            });
            profile::time(6, || match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    gate_up_silu(gate, up, &s.xb, &s.xq, &mut s.gate);
                    s.xq.quantize(&s.gate, c.ffn_dim);
                    down.matmul(&s.gate, &s.xq, &mut s.proj);
                    for (x, f) in s.x.iter_mut().zip(&s.proj) {
                        *x += f;
                    }
                }
                Ffn::Moe { router, bias, experts, shared, shared_gate } => {
                    if c.mla.is_some() {
                        moe_route_deepseek(c, router, bias.as_deref(), &s.xb, &mut s.picks, &mut s.moe);
                    } else {
                        moe_route(router, c.n_experts_used, &s.xb, dim, &mut s.picks, &mut s.moe);
                    }
                    moe_experts(experts.len(), |e| experts[e], &s.picks, c.n_experts_used, &s.xb, dim, &mut s.x, &mut s.moe);
                    if let Some(sh) = shared {
                        let hid = sh.gate.rows;
                        let h = &mut s.gate[..n * hid];
                        gate_up_silu(&sh.gate, &sh.up, &s.xb, &s.xq, h);
                        s.xq.quantize(h, hid);
                        sh.down.matmul(h, &s.xq, &mut s.proj);
                        for ((x, f), xb) in s.x.chunks_exact_mut(dim).zip(s.proj.chunks_exact(dim)).zip(s.xb.chunks_exact(dim)) {
                            let w = shared_gate.as_ref().map_or(1.0, |g| 1.0 / (1.0 + (-dot_f32(g, xb)).exp()));
                            for (x, f) in x.iter_mut().zip(f) {
                                *x += w * f;
                            }
                        }
                    }
                }
            });
        }

        if want_logits {
            profile::time(7, || {
                let last = &s.x[(n - 1) * dim..n * dim];
                let mut normed = vec![0f32; dim];
                rms_norm_rows(&mut normed, last, &self.output_norm, c.norm_eps);
                s.xq.quantize(&normed, dim);
                self.output.matmul(&normed, &s.xq, &mut s.logits);
            });
        }
        s.pos += n;
        Ok(())
    }

    /// Latent attention's inputs for the batch: the queries (each head's position-free part moved into the
    /// latent space, then its RoPE part) into `s.qa`, and the cache rows (normed latent + RoPE key; the
    /// values are the latent alone).
    #[allow(clippy::too_many_arguments)]
    fn mla_in(&self, s: &mut Session, n: usize, k_base: usize, v_base: usize, wq_a: Option<&(Matrix, Vec<f32>)>, wq: &Matrix, wkv_a: &Matrix, kv_a_norm: &[f32], k_b: &[Matrix]) {
        let c = &self.cfg;
        let (m, hd, rd, pos0) = (c.mla.unwrap(), c.head_dim, c.rope_dim, s.pos);
        let (kd, vd) = (c.k_row(), c.v_row());
        match wq_a {
            Some((wa, norm)) => {
                matmul_many(&mut [(wa, &mut s.lat), (wkv_a, &mut s.k)], &s.xb, &s.xq);
                rms_norm_inplace(&mut s.lat, norm, c.norm_eps);
                s.xq.quantize(&s.lat, m.q_lora);
                wq.matmul(&s.lat, &s.xq, &mut s.q);
            }
            None => matmul_many(&mut [(wq, &mut s.q), (wkv_a, &mut s.k)], &s.xb, &s.xq),
        }
        for b in 0..n {
            let (lat, pe) = s.k[b * kd..(b + 1) * kd].split_at_mut(m.kv_lora);
            rms_norm_inplace(lat, kv_a_norm, c.norm_eps);
            rope(pe, rd, rd, pos0 + b, &s.rope_inv_freq, c.rope_neox);
            for h in 0..c.n_heads {
                let q0 = (b * c.n_heads + h) * hd + m.nope;
                rope(&mut s.q[q0..q0 + rd], rd, rd, pos0 + b, &s.rope_inv_freq, c.rope_neox);
            }
            let (ka, va) = (k_base + (pos0 + b) * kd, v_base + (pos0 + b) * vd);
            s.k_cache[ka..ka + kd].copy_from_slice(&s.k[b * kd..(b + 1) * kd]);
            s.v_cache[va..va + vd].copy_from_slice(&s.k[b * kd..b * kd + vd]);
        }
        for (h, kb) in k_b.iter().enumerate() {
            s.head_in.clear();
            for b in 0..n {
                let q0 = (b * c.n_heads + h) * hd;
                s.head_in.extend_from_slice(&s.q[q0..q0 + m.nope]);
            }
            s.xq.quantize(&s.head_in, m.nope);
            s.head_out.resize(n * m.kv_lora, 0.0);
            kb.matmul(&s.head_in, &s.xq, &mut s.head_out);
            for b in 0..n {
                let (a0, q0) = ((b * c.n_heads + h) * kd, (b * c.n_heads + h) * hd + m.nope);
                s.qa[a0..a0 + m.kv_lora].copy_from_slice(&s.head_out[b * m.kv_lora..(b + 1) * m.kv_lora]);
                s.qa[a0 + m.kv_lora..a0 + kd].copy_from_slice(&s.q[q0..q0 + rd]);
            }
        }
    }

    /// A Gated DeltaNet layer over the batch, token by token through its state: the outputs (each value
    /// head's, after the gated norm) go to `s.att_out`, ready for ssm_out.
    fn delta_net(&self, d: &DeltaLayer, s: &mut Session, l: usize, n: usize) {
        let c = &self.cfg;
        let dc = c.delta.unwrap();
        let (kd, vd, cd, hk) = (dc.key_dim(), dc.value_dim(), dc.conv_dim(), dc.head);
        match &d.wba[..] {
            [wb, wa] => {
                let (bb, aa) = s.dba[..2 * n * dc.v_heads].split_at_mut(n * dc.v_heads);
                matmul_many(&mut [(&d.wqkv, &mut s.dqkv[..]), (&d.wz, &mut s.dz[..]), (wb, bb), (wa, aa)], &s.xb, &s.xq);
            }
            _ => matmul_many(&mut [(&d.wqkv, &mut s.dqkv[..]), (&d.wz, &mut s.dz[..]), (&d.wba[0], &mut s.dba[..])], &s.xb, &s.xq),
        }
        let slot = c.slot(l);
        let taps = dc.conv - 1;
        let conv_state = &mut s.conv_state[slot * dc.conv_len()..(slot + 1) * dc.conv_len()];
        let state = &mut s.ssm_state[slot * dc.state_len()..(slot + 1) * dc.state_len()];
        let (eps, scale) = (c.norm_eps, 1.0 / (hk as f32).sqrt());
        for b in 0..n {
            // causal convolution of each channel over its last inputs, then SiLU
            let x = &mut s.dqkv[b * cd..(b + 1) * cd];
            for ((v, w), st) in x.iter_mut().zip(d.conv.chunks_exact(dc.conv)).zip(conv_state.chunks_exact_mut(taps)) {
                let mut acc = w[taps] * *v;
                for (wk, sv) in w.iter().zip(st.iter()) {
                    acc += wk * sv;
                }
                st.copy_within(1.., 0);
                st[taps - 1] = *v;
                *v = acc / (1.0 + (-acc).exp());
            }
            // unit-length query and key heads (as ggml's l2_norm: x / max(|x|, eps)), the query scaled
            let (q, rest) = x.split_at_mut(kd);
            let (k, v) = rest.split_at_mut(kd);
            for (h, f) in q.chunks_exact_mut(hk).map(|h| (h, scale)).chain(k.chunks_exact_mut(hk).map(|h| (h, 1.0))) {
                let inv = f / dot_f32(h, h).sqrt().max(eps);
                h.iter_mut().for_each(|e| *e *= inv);
            }
            let (q, k, v) = (&*q, &*k, &*v);
            let ba = &s.dba;
            let z = &s.dz[b * vd..(b + 1) * vd];
            let out = &mut s.att_out[b * vd..(b + 1) * vd];
            state.par_chunks_mut(hk * hk).zip(out.par_chunks_mut(hk)).enumerate().for_each(|(h, (st, o))| {
                let (g, (bi, ai)) = (dc.key_head(h), dc.ba_at(n, b, h));
                let beta = 1.0 / (1.0 + (-ba[bi]).exp());
                let a = ba[ai] + d.dt_bias[h];
                let softplus = if a > 20.0 { a } else { a.exp().ln_1p() };
                let decay = (softplus * d.a[h]).exp();
                let (qh, kh, vh) = (&q[g * hk..(g + 1) * hk], &k[g * hk..(g + 1) * hk], &v[h * hk..(h + 1) * hk]);
                // each value column of the state: decay, correct what it recalls for k towards v, read with q
                for ((col, ov), vv) in st.chunks_exact_mut(hk).zip(o.iter_mut()).zip(vh) {
                    col.iter_mut().for_each(|e| *e *= decay);
                    let delta = (vv - dot_f32(col, kh)) * beta;
                    for (e, kv) in col.iter_mut().zip(kh) {
                        *e += kv * delta;
                    }
                    *ov = dot_f32(col, qh);
                }
                // gated RMSNorm: norm(o) * w * silu(z)
                let inv = 1.0 / (dot_f32(o, o) / hk as f32 + eps).sqrt();
                for ((ov, w), zv) in o.iter_mut().zip(&d.norm).zip(&z[h * hk..(h + 1) * hk]) {
                    *ov *= inv * w * zv / (1.0 + (-zv).exp());
                }
            });
        }
    }

    /// Causal multi-head attention for the batch, reading keys/values from the cache. Each kv head has
    /// `kd`-wide keys and `vd`-wide values (equal, except for latent attention).
    fn attention(&self, s: &mut Session, k_base: usize, v_base: usize, n: usize) {
        let c = &self.cfg;
        let (kd, vd, scale) = c.att_dims();
        let (k_row, v_row) = (c.k_row(), c.v_row());
        let (q_dim, o_dim) = (c.n_heads * kd, c.n_heads * vd);
        let group = c.n_heads / c.n_kv_heads;
        let pos0 = s.pos;
        let k_cache = &s.k_cache[k_base..k_base + (pos0 + n) * k_row];
        let v_cache = &s.v_cache[v_base..v_base + (pos0 + n) * v_row];
        let q = if c.mla.is_some() { &s.qa } else { &s.q };
        let span_of = |b: usize| pos0 + b + 1; // causal: token b sees positions 0..=pos0+b

        // Work unit = (token, kv head, slice of the context). The `group` query heads sharing a kv head
        // are adjacent in q and att_out, so each key/value row is read once for all of them. When there
        // are fewer (token, kv head) pairs than threads (single-token decoding), the context is split into
        // slices whose partial softmaxes are merged afterwards ("flash-decoding").
        let pairs = n * c.n_kv_heads;
        let threads = rayon::current_num_threads();
        let splits = if pairs >= threads { 1 } else { threads.div_ceil(pairs).min(span_of(0).div_ceil(64)).max(1) };
        let unit = group * vd + 2 * group; // per slice: unnormalised outputs, then max and sum per head
        s.att_partial.resize(pairs * splits * unit, 0.0);

        let slice_task = |scores: &mut Vec<f32>, (idx, part): (usize, &mut [f32])| {
            let (pair, split) = (idx / splits, idx % splits);
            let (b, kvh) = (pair / c.n_kv_heads, pair % c.n_kv_heads);
            let span = span_of(b);
            let per = span.div_ceil(splits);
            let (t0, t1) = ((split * per).min(span), ((split + 1) * per).min(span));
            let len = t1 - t0;
            let q0 = b * q_dim + kvh * group * kd;
            let qs = &q[q0..q0 + group * kd];
            let (out, stats) = part.split_at_mut(group * vd);
            out.fill(0.0);
            if len == 0 {
                for g in 0..group {
                    stats[2 * g] = f32::NEG_INFINITY;
                    stats[2 * g + 1] = 0.0;
                }
                return;
            }
            scores.resize(group * len, 0.0);
            for t in t0..t1 {
                let kt = &k_cache[t * k_row + kvh * kd..t * k_row + (kvh + 1) * kd];
                for (g, qh) in qs.chunks_exact(kd).enumerate() {
                    scores[g * len + (t - t0)] = dot_f32(qh, kt) * scale;
                }
            }
            for (g, row) in scores.chunks_exact_mut(len).enumerate() {
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut total = 0.0;
                for sc in row.iter_mut() {
                    *sc = (*sc - max).exp();
                    total += *sc;
                }
                stats[2 * g] = max;
                stats[2 * g + 1] = total;
            }
            for t in t0..t1 {
                let vt = &v_cache[t * v_row + kvh * vd..t * v_row + (kvh + 1) * vd];
                for (g, oh) in out.chunks_exact_mut(vd).enumerate() {
                    let w = scores[g * len + (t - t0)];
                    for (o, v) in oh.iter_mut().zip(vt) {
                        *o += w * v;
                    }
                }
            }
        };
        // a single token over a short context is too little work to be worth waking threads
        if n * c.n_heads * (pos0 + n) * kd < 1 << 16 {
            let mut scores = Vec::new();
            s.att_partial.chunks_mut(unit).enumerate().for_each(|t| slice_task(&mut scores, t));
        } else {
            s.att_partial.par_chunks_mut(unit).enumerate().for_each_init(Vec::new, slice_task);
        }

        // merge the slices of each (token, kv head) into normalised head outputs
        for pair in 0..pairs {
            let (b, kvh) = (pair / c.n_kv_heads, pair % c.n_kv_heads);
            let parts = &s.att_partial[pair * splits * unit..(pair + 1) * splits * unit];
            for g in 0..group {
                let m = parts.chunks_exact(unit).map(|p| p[group * vd + 2 * g]).fold(f32::NEG_INFINITY, f32::max);
                let o0 = b * o_dim + (kvh * group + g) * vd;
                let out = &mut s.att_out[o0..o0 + vd];
                out.fill(0.0);
                let mut total = 0.0;
                for p in parts.chunks_exact(unit) {
                    let w = (p[group * vd + 2 * g] - m).exp();
                    if w == 0.0 {
                        continue;
                    }
                    total += p[group * vd + 2 * g + 1] * w;
                    for (o, v) in out.iter_mut().zip(&p[g * vd..(g + 1) * vd]) {
                        *o += w * v;
                    }
                }
                let inv = 1.0 / total;
                out.iter_mut().for_each(|o| *o *= inv);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::route;

    #[test]
    fn route_keeps_top_k_with_softmax_over_them() {
        let logits = [0.5f32, 2.0, -1.0, 2.0, 1.0];
        let mut picks = Vec::new();
        route(&logits, 3, &mut picks);
        // ties go to the lower expert index
        assert_eq!(picks.iter().map(|p| p.0).collect::<Vec<_>>(), vec![1, 3, 4]);
        let z = 2.0f32.exp() * 2.0 + 1.0f32.exp();
        for (p, want) in picks.iter().zip([2.0f32.exp() / z, 2.0f32.exp() / z, 1.0f32.exp() / z]) {
            assert!((p.1 - want).abs() < 1e-6, "{p:?} vs {want}");
        }
    }
}
