//! NVIDIA GPU backend. Loads the CUDA driver (nvcuda.dll) and NVRTC at runtime, compiles the kernels
//! below to PTX for the installed GPU, and runs the whole transformer on the device. No build-time CUDA
//! dependency: without a GPU the binary still works on the CPU.

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use crate::gguf::{GgmlType, Gguf, TensorInfo};
use crate::model::Config;
use crate::profile;
use crate::quant::f16_to_f32;

const MAX_BATCH: usize = 1024; // prompt tokens per GPU pass: big batches keep the attention kernel busy

/// Prompt tokens per GPU pass for this model. Qwen3-Next's experts (an 80B mixture) never fit in GPU memory or
/// even RAM: every pass streams nearly all of them, so a pass of 4096 tokens reads a long prompt's experts once
/// where 1024 would read them several times (for about 400 MB more activations).
fn max_batch(cfg: &Config) -> usize {
    let b = if cfg.delta.is_some() { 4096 } else { MAX_BATCH };
    match BATCH_CAP.load(Ordering::Relaxed) {
        0 => b,
        cap => b.min(cap),
    }
}
/// A smaller prompt batch fit_ctx chose when the weights leave little memory (0: none). Only ever lowered: a
/// session made before keeps scratch for the bigger batch and just runs its prompts in smaller pieces.
static BATCH_CAP: AtomicUsize = AtomicUsize::new(0);
/// The smallest prompt batch fit_ctx goes down to for more context.
const MIN_FIT_BATCH: usize = 256;
/// fit_ctx makes prompt batches smaller (down to MIN_FIT_BATCH) while the context is shorter than this.
const FIT_SHORT_CTX: usize = 8192;
const MATMUL_TB: usize = 8; // tokens per matmul block row (must match TB in the kernel source)
/// Rows per warp in the K-quant generation matmuls (passed to the kernels as -DRPW).
/// GHOST_INFER_RPW overrides it for tuning.
fn kq_rpw() -> usize {
    static RPW: OnceLock<usize> = OnceLock::new();
    *RPW.get_or_init(|| std::env::var("GHOST_INFER_RPW").ok().and_then(|v| v.parse().ok()).filter(|&v| (1..=16).contains(&v)).unwrap_or(1))
}
/// Rows per warp in the K-quant decode-batch matmuls (speculative verify; -DDP4A_BATCH_RPW).
fn batch_rpw() -> usize {
    static R: OnceLock<usize> = OnceLock::new();
    *R.get_or_init(|| std::env::var("GHOST_INFER_BATCH_RPW").ok().and_then(|v| v.parse().ok()).filter(|&v| (1..=8).contains(&v)).unwrap_or(2))
}

/// Activation values per int8 scale in the K-quant prompt matmuls (passed as -DQCH): 128 or 256.
/// 128 is ~1% slower than 256 and about halves the log-probability drift; GHOST_INFER_QCH=256 overrides.
fn qch() -> usize {
    static QCH: OnceLock<usize> = OnceLock::new();
    *QCH.get_or_init(|| std::env::var("GHOST_INFER_QCH").ok().and_then(|v| v.parse().ok()).filter(|&v| v == 128 || v == 256).unwrap_or(128))
}
/// KV cache in int8 with a scale per (position, kv head) instead of f16: about half the memory per
/// token. Process-wide, because the kernels are compiled for one or the other (-DKV8); set it (--kv)
/// before the first model loads. Default int8; GHOST_INFER_KV=f16 changes the default.
static KV_INT8: OnceLock<bool> = OnceLock::new();
pub fn set_kv_int8(on: bool) {
    let _ = KV_INT8.set(on);
}
fn kv_int8() -> bool {
    *KV_INT8.get_or_init(|| std::env::var("GHOST_INFER_KV").map_or(true, |v| v != "f16"))
}
/// Bytes per cached k or v value.
fn kv_bytes() -> usize {
    if kv_int8() { 1 } else { 2 }
}
const KQ_TB: usize = 4; // tokens per block row in the K-quant generation matmuls (must match TBK)
const TILE: usize = 64; // must match TILE in the kernel source
const TILED_MIN_BATCH: usize = 16; // from this many tokens, the int8 tiled matmul beats one-warp-per-row
const ATT_MAX_CHUNK: usize = 256; // must match ATT_MAX_CHUNK
/// (token, head, slice) rows of scores and weights the tiled verify attention holds per head (s.att_scr): the
/// tokens of one launch times their slices.
const AV_SCR_ROWS: usize = 1024;
/// A model whose verify batches go through the tiled attention (and whose sessions keep its scratch).
fn vfy_attention_fits(c: &Config) -> bool {
    c.mla.is_none() && !c.is_moe() && c.head_dim == 128 && c.n_heads % c.n_kv_heads == 0 && c.n_heads / c.n_kv_heads <= 8 && kv_int8()
}
const ATT_MAX_HD: usize = 128; // must match ATT_MAX_HD
const ATT_MAX_G: usize = 8; // query heads per kv head the GQA attention kernel handles (must match ATT_MAX_G)
const SPLIT_MIN_SPAN: usize = 32; // contexts up to this many positions are not split across blocks (must match ATT_SPLIT_MIN_SPAN)
/// Most positions per attention block once a context is split (GHOST_INFER_ATT_CAP, 32..ATT_MAX_CHUNK): shorter
/// slices mean more blocks and shorter sequential sums. Qwen3 4B on an RTX 2060 Super, tok/s at 1K / 2K / 4K /
/// 8K positions: 256 (up to this commit) - / 94.6 / - / 67.5, 160 110.5 / 101.7 / 87.4 / 71.0, 128 110.3 / 103.3 /
/// 88.8 / 71.6, 96 110.7 / 99.0 / 86.3 / 70.2, 64 - / 100.1 / - / 67.7.
fn att_chunk_cap() -> usize {
    static C: OnceLock<usize> = OnceLock::new();
    *C.get_or_init(|| std::env::var("GHOST_INFER_ATT_CAP").ok().and_then(|v| v.parse().ok()).unwrap_or(128).clamp(32, ATT_MAX_CHUNK))
}
const ATT_TARGET_BLOCKS: usize = 96; // aim for a few blocks per SM when splitting attention for one token
const PREFILL_QT: usize = 32; // query tokens per prompt-attention block (must match PQT)
const PREFILL_KT: usize = 32; // key positions per shared-memory tile (must match PKT)
const PREFILL_TC_QT: usize = 64; // query tokens per tensor-core prompt-attention block (must match BQ)
const SPLIT_MAX_BATCH: usize = TILED_MIN_BATCH - 1; // tokens per split-attention launch
// latent attention: batches up to this many tokens split each token's context into slices of about
// MLA_SPLIT_SPAN positions (at most MLA_MAX_SPLIT); bigger batches have blocks enough without
const MLA_SPLIT_MAX_BATCH: usize = 4;
const MLA_SPLIT_SPAN: usize = 32;
const MLA_MAX_SPLIT: usize = 128;
const MLA_T: usize = 32; // must match MLA_T
const HMV_T: usize = 8; // must match HMV_T

/// head_matvec's row slices per head: one token's heads alone are too few blocks to fill the GPU.
fn hmv_slices(n: usize, rows: usize) -> usize {
    if n > 2 * HMV_T { 1 } else { (rows / 64).max(1) }
}
/// Most tokens per decode batch: one token, or a speculative verify batch.
pub const DECODE_MAX_BATCH: usize = 8;
/// Most tokens per verify batch on the tensor-core verify kernels (GpuModel::verify_max says whether a
/// model can use them).
pub const VERIFY_MAX_BATCH: usize = 64;
/// Verify batches of at least this many tokens take the tensor-core kernels (GHOST_INFER_VFY_MIN tunes it).
fn vfy_min() -> usize {
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| std::env::var("GHOST_INFER_VFY_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(3).max(2))
}

const KERNELS: &str = concat!(include_str!("iq_tables.cu"), r#"
#define TB 8
#define NEG_INF __int_as_float(0xff800000) // NVRTC has no <math.h> INFINITY
extern "C" {

__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__device__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    return v;
}
// block-wide reductions; blockDim.x must be a multiple of 32
__device__ float block_sum(float v, float* sh) {
    v = warp_sum(v);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? sh[threadIdx.x] : 0.f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}
__device__ float block_max(float v, float* sh) {
    v = warp_max(v);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    v = (threadIdx.x < (blockDim.x >> 5)) ? sh[threadIdx.x] : NEG_INF;
    if (wid == 0) v = warp_max(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

// ---- weight formats (repacked at upload so every field is naturally aligned) ----
// fmt 0, Q8_0: int8 values (rows x cols) + a separate f32 scale per 32
// fmt 1, Q4_K: the GGUF block as is, 144 bytes per 256: f16 d, f16 dmin, 12 bytes of 6-bit scales/mins,
//              128 bytes of nibbles
// fmt 2, Q6_K: 208-byte blocks of 256 (128 bytes low nibbles, 64 bytes 2-bit highs, 16 int8 scales), and
//              the f32 d of each block in a separate array (wd)
// Both block sizes are multiples of 16 bytes, so the quant bytes can be read with 128-bit loads.
#define Q4K_BYTES 144
#define Q6K_BYTES 208

__device__ __forceinline__ float h2f(unsigned int h) {
    float f;
    asm("{.reg .b16 t; cvt.u16.u32 t, %1; cvt.f32.f16 %0, t;}" : "=f"(f) : "r"(h));
    return f;
}
// f32 -> f16: the f16 KV cache and tensor-core operands; attention arithmetic stays in f32.
__device__ __forceinline__ unsigned short f2h(float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
// 4 f16 values (one 8-byte load) to floats
__device__ __forceinline__ float4 h4f(const unsigned short* p) {
    uint2 w = *reinterpret_cast<const uint2*>(p);
    return make_float4(h2f(w.x & 0xFFFF), h2f(w.x >> 16), h2f(w.y & 0xFFFF), h2f(w.y >> 16));
}

// KV cache element. -DKV8: int8 with one f32 scale per (position, kv head) in ks / vs (symmetric,
// quantised as each row is written, like LMDeploy's online int8 cache; a scale per 32 values, as
// llama.cpp's q8_0 cache, measured no closer to f16 here and costs 9% more memory). Otherwise f16 and
// ks / vs unused. The kv_load* helpers return the stored values; KV_SCALE gives the factor that turns
// them into floats.
#ifdef KV8
typedef signed char kv_t;
#define KV_SCALE(sc, pos, n_kv, kvh) ((sc)[(size_t)(pos) * (n_kv) + (kvh)])
__device__ __forceinline__ float kv_load1(const kv_t* p) { return (float)*p; }
__device__ __forceinline__ float2 kv_load2(const kv_t* p) {
    int w = *reinterpret_cast<const short*>(p);
    return make_float2((float)(signed char)w, (float)(w >> 8));
}
__device__ __forceinline__ float4 kv_load4(const kv_t* p) {
    int w = *reinterpret_cast<const int*>(p);
    return make_float4((float)(signed char)w, (float)(signed char)(w >> 8), (float)(signed char)(w >> 16), (float)(w >> 24));
}
#else
typedef unsigned short kv_t;
#define KV_SCALE(sc, pos, n_kv, kvh) 1.f
__device__ __forceinline__ float kv_load1(const kv_t* p) { return h2f(*p); }
__device__ __forceinline__ float2 kv_load2(const kv_t* p) {
    unsigned w = *reinterpret_cast<const unsigned*>(p);
    return make_float2(h2f(w & 0xFFFF), h2f(w >> 16));
}
__device__ __forceinline__ float4 kv_load4(const kv_t* p) { return h4f(p); }
#endif

// One head's row of hd floats into the cache; the whole block calls it (int8 reduces the row's max).
__device__ __forceinline__ void kv_store(kv_t* __restrict__ dst, float* __restrict__ scale, const float* __restrict__ src, int hd, float* sh) {
#ifdef KV8
    float amax = 0.f;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) amax = fmaxf(amax, fabsf(src[d]));
    amax = block_max(amax, sh);
    float dd = amax / 127.f, inv = dd != 0.f ? 1.f / dd : 0.f;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) dst[d] = (signed char)__float2int_rn(src[d] * inv);
    if (threadIdx.x == 0) *scale = dd;
#else
    for (int d = threadIdx.x; d < hd; d += blockDim.x) dst[d] = f2h(src[d]);
#endif
}
// d and dmin of a Q4_K block (one 32-bit load)
__device__ __forceinline__ void q4k_dm(const unsigned char* b, float* d, float* dmin) {
    unsigned int w = *reinterpret_cast<const unsigned int*>(b);
    *d = h2f(w & 0xFFFF);
    *dmin = h2f(w >> 16);
}

__device__ __forceinline__ void q4k_scale_min(int j, const unsigned char* s, int* sc, int* m) {
    if (j < 4) { *sc = s[j] & 63; *m = s[j + 4] & 63; }
    else { *sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4); *m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4); }
}

// Same, for sub-blocks j and j+1 (j even), from the 12 scale bytes as three words.
__device__ __forceinline__ void q4k_scale_min2w(int j, unsigned int w0, unsigned int w1, unsigned int w2, int* sc1, int* m1, int* sc2, int* m2) {
    #define SB(i) ((int)(((i) < 4 ? w0 : (i) < 8 ? w1 : w2) >> (8 * ((i) & 3)) & 0xFF))
    if (j < 4) {
        *sc1 = SB(j) & 63;            *m1 = SB(j + 4) & 63;
        *sc2 = SB(j + 1) & 63;        *m2 = SB(j + 5) & 63;
    } else {
        *sc1 = (SB(j + 4) & 0xF) | ((SB(j - 4) >> 6) << 4); *m1 = (SB(j + 4) >> 4) | ((SB(j) >> 6) << 4);
        *sc2 = (SB(j + 5) & 0xF) | ((SB(j - 3) >> 6) << 4); *m2 = (SB(j + 5) >> 4) | ((SB(j + 1) >> 6) << 4);
    }
    #undef SB
}

// value i of a Q4_K / Q6_K row
__device__ float deq_q4k(const unsigned char* row, int i) {
    const unsigned char* b = row + (size_t)(i >> 8) * Q4K_BYTES;
    int k = i & 255, j2 = k >> 6, half = (k >> 5) & 1, l = k & 31;
    int sc, m;
    q4k_scale_min(2 * j2 + half, b + 4, &sc, &m);
    int q = b[16 + 32 * j2 + l];
    q = half ? (q >> 4) : (q & 15);
    float d, dmin;
    q4k_dm(b, &d, &dmin);
    return d * sc * q - dmin * m;
}
__device__ float deq_q6k(const unsigned char* row, const float* drow, int i) {
    const unsigned char* b = row + (size_t)(i >> 8) * Q6K_BYTES;
    int k = i & 255, half = k >> 7, g = (k >> 5) & 3, l = k & 31;
    int lo = b[64 * half + (g & 1) * 32 + l];
    lo = (g >= 2) ? (lo >> 4) : (lo & 15);
    int hi = (b[128 + 32 * half + l] >> (2 * g)) & 3;
    return drow[i >> 8] * (float)(signed char)b[192 + 8 * half + 2 * g + (l >> 4)] * ((lo | (hi << 4)) - 32);
}

// x[b] = embedding row of tokens[b], any weight format
__global__ void embed(int fmt, const unsigned char* __restrict__ w, const float* __restrict__ wd,
                      const int* __restrict__ tokens, float* __restrict__ x, int cols) {
    int b = blockIdx.x;
    size_t tok = tokens[b];
    float* o = x + (size_t)b * cols;
    if (fmt == 0) {
        for (int i = threadIdx.x; i < cols; i += blockDim.x)
            o[i] = wd[tok * (cols >> 5) + (i >> 5)] * (float)(signed char)w[tok * cols + i];
    } else {
        size_t row_bytes = (size_t)(cols >> 8) * (fmt == 1 ? Q4K_BYTES : Q6K_BYTES);
        const unsigned char* row = w + tok * row_bytes;
        const float* drow = wd + tok * (cols >> 8); // Q6_K
        for (int i = threadIdx.x; i < cols; i += blockDim.x) o[i] = fmt == 1 ? deq_q4k(row, i) : deq_q6k(row, drow, i);
    }
}

// Per head of the packed [q | k | v] rows: RMSNorm of q and k (Qwen3; skipped when qn is null), RoPE on
// q and k, and k, v copied into the cache. grid (tokens, n_heads + 2 * n_kv), block = 64.
// neox: pairs (x[i], x[i + rope_dim/2]) (Qwen); otherwise adjacent pairs (x[2i], x[2i+1]) (GGUF llama).
__global__ void qk_norm_rope(float* __restrict__ qkv, const float* __restrict__ qn, const float* __restrict__ kn,
                             kv_t* __restrict__ kc, kv_t* __restrict__ vc, float* __restrict__ ks, float* __restrict__ vs,
                             int pos0, const int* __restrict__ posp, int n_heads,
                             int n_kv, int hd, int rope_dim, float base, const float* __restrict__ ff, int neox, float eps) {
    __shared__ float sh[32];
    if (posp) pos0 = *posp; // CUDA graphs: the position comes from device memory
    int b = blockIdx.x, h = blockIdx.y, pos = pos0 + b;
    int q_dim = n_heads * hd, kv_dim = n_kv * hd;
    float* row = qkv + (size_t)b * (q_dim + 2 * kv_dim);
    if (h >= n_heads + n_kv) { // a v head: straight into the cache (same for the whole block)
        int vh = h - n_heads - n_kv;
        kv_store(vc + (size_t)pos * kv_dim + vh * hd, vs + (size_t)pos * n_kv + vh, row + q_dim + kv_dim + vh * hd, hd, sh);
        return;
    }
    float* vec = row + h * hd; // k heads directly follow the q heads
    if (qn) {
        const float* w = h < n_heads ? qn : kn;
        float ss = 0.f;
        for (int d = threadIdx.x; d < hd; d += blockDim.x) ss += vec[d] * vec[d];
        ss = block_sum(ss, sh);
        float scale = rsqrtf(ss / hd + eps);
        for (int d = threadIdx.x; d < hd; d += blockDim.x) vec[d] = vec[d] * scale * w[d];
        __syncthreads();
    }
    int half = rope_dim / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = powf(base, -2.0f * (float)i / (float)rope_dim);
        if (ff) freq /= ff[i]; // Llama 3 frequency scaling
        float s, c;
        sincosf((float)pos * freq, &s, &c);
        int ia = neox ? i : 2 * i, ib = neox ? i + half : 2 * i + 1;
        float a = vec[ia], bb = vec[ib];
        vec[ia] = a * c - bb * s;
        vec[ib] = a * s + bb * c;
    }
    if (h >= n_heads) {
        __syncthreads();
        int kh = h - n_heads;
        kv_store(kc + (size_t)pos * kv_dim + kh * hd, ks + (size_t)pos * n_kv + kh, vec, hd, sh);
    }
}

// int8 quantisation (as quantize_q8) of 32-value blocks held 4 per lane by 8 adjacent lanes; the whole
// warp must call it. Lane i of the row's float4s writes its 4 bytes; the block's first lane its scale and sum.
__device__ __forceinline__ void quant8x4(float4 v, bool ok, int i, signed char* __restrict__ q, float* __restrict__ d, int* __restrict__ sums) {
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    for (int o = 1; o < 8; o <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    float dd = amax / 127.f;
    float inv = dd != 0.f ? 1.f / dd : 0.f;
    int q0 = (int)roundf(v.x * inv), q1 = (int)roundf(v.y * inv), q2 = (int)roundf(v.z * inv), q3 = (int)roundf(v.w * inv);
    int s = q0 + q1 + q2 + q3;
    for (int o = 1; o < 8; o <<= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (ok) {
        *reinterpret_cast<int*>(q + 4 * (size_t)i) = (q0 & 255) | ((q1 & 255) << 8) | ((q2 & 255) << 16) | ((unsigned)q3 << 24);
        if ((i & 7) == 0) { d[i >> 3] = dd; sums[i >> 3] = s; }
    }
}

// one block per row
__global__ void rmsnorm(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out,
                        int cols, float eps) {
    __shared__ float sh[32];
    const float* xr = x + (size_t)blockIdx.x * cols;
    float* o = out + (size_t)blockIdx.x * cols;
    float ss = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, sh);
    float scale = rsqrtf(ss / cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) o[i] = xr[i] * scale * w[i];
}

// RMSNorm straight to int8 activations (one token per block, up to 1024 threads, cols a multiple of 32):
// the one-token K-quant matmuls take xq / xd directly, so there is no f32 copy and no quantize launch.
__global__ void rmsnorm_q8(const float* __restrict__ x, const float* __restrict__ w, signed char* __restrict__ xq,
                           float* __restrict__ xd, int* __restrict__ xsum, int cols, float eps) {
    __shared__ float sh[32];
    const float4* xr = reinterpret_cast<const float4*>(x + (size_t)blockIdx.x * cols);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    int nv = cols >> 2;
    float ss = 0.f;
    for (int i = threadIdx.x; i < nv; i += blockDim.x) { float4 v = xr[i]; ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w; }
    ss = block_sum(ss, sh);
    float scale = rsqrtf(ss / cols + eps);
    size_t r = blockIdx.x;
    for (int base = 0; base < nv; base += blockDim.x) { // every lane takes part in the shuffles
        int i = base + threadIdx.x;
        bool ok = i < nv;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (ok) {
            float4 a = xr[i], g = w4[i];
            v = make_float4(a.x * scale * g.x, a.y * scale * g.y, a.z * scale * g.z, a.w * scale * g.w);
        }
        quant8x4(v, ok, i, xq + r * cols, xd + r * (cols >> 5), xsum + r * (cols >> 5));
    }
}

// Generation matmuls: out[b*out_stride + out_off + r] (+)= dot(W[r], x[b]), f32 activations.
// One warp per row, TB tokens per block in y.

// Q8_0: W is int8 (rows x cols) with one f32 scale per 32 values.
__global__ void matmul_q8(const signed char* __restrict__ wq, const float* __restrict__ wd,
                          const float* __restrict__ x, float* __restrict__ out,
                          int rows, int cols, int n, int accumulate, int out_stride, int out_off) {
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    int b0 = blockIdx.y * TB;
    int nb = min(TB, n - b0);
    float acc[TB];
    #pragma unroll
    for (int j = 0; j < TB; j++) acc[j] = 0.f;
    const signed char* wr = wq + (size_t)row * cols;
    const float* dr = wd + (size_t)row * (cols >> 5);
    for (int i = lane * 4; i < cols; i += 128) {
        int packed = *reinterpret_cast<const int*>(wr + i);
        float d = dr[i >> 5];
        float w0 = d * (float)(signed char)(packed);
        float w1 = d * (float)(signed char)(packed >> 8);
        float w2 = d * (float)(signed char)(packed >> 16);
        float w3 = d * (float)(signed char)(packed >> 24);
        #pragma unroll
        for (int j = 0; j < TB; j++) {
            if (j < nb) {
                float4 xv = *reinterpret_cast<const float4*>(x + (size_t)(b0 + j) * cols + i);
                acc[j] += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < TB; j++) {
        float v = warp_sum(acc[j]);
        if (lane == 0 && j < nb) {
            size_t idx = (size_t)(b0 + j) * out_stride + out_off + row;
            out[idx] = accumulate ? out[idx] + v : v;
        }
    }
}

// K-quant generation matmuls: each warp handles RPW rows so the activations a lane needs are loaded
// once into registers and reused for all of them (4-bit weights are ~7x smaller than f32 activations,
// so re-reading the activations per row would dominate). TBK tokens per block in y.
// Compiled twice: TBK = 1 for generation (few registers, so many warps keep loads in flight) and
// TBK = 4 for short multi-token batches. RPW (rows per warp) comes from the compile options.

}  // extern "C": the templates below need C++ linkage

// Q4_K: per super-block each lane takes 4 bytes of a 64-value chunk, i.e. 4 values of two sub-blocks.
template <int TBK>
__device__ void q4k_matvec(const unsigned char* __restrict__ w, const float* __restrict__ x, float* __restrict__ out,
                           int rows, int cols, int n, int accumulate, int out_stride, int out_off) {
    int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * RPW;
    int lane = threadIdx.x & 31;
    if (row0 >= rows) return;
    int b0 = blockIdx.y * TBK, nb = min(TBK, n - b0);
    int nsb = cols >> 8;
    int j2 = lane >> 3, sub = lane & 7;
    size_t row_bytes = (size_t)nsb * Q4K_BYTES;
    float acc[RPW][TBK];
    #pragma unroll
    for (int r = 0; r < RPW; r++)
        #pragma unroll
        for (int j = 0; j < TBK; j++) acc[r][j] = 0.f;
    #pragma unroll 4 // several super-blocks' loads in flight before any is used
    for (int sb = 0; sb < nsb; sb++) {
        int base = sb * 256 + 64 * j2 + 4 * sub;
        float4 xl[TBK], xh[TBK];
        float sl[TBK], sh[TBK];
        #pragma unroll
        for (int j = 0; j < TBK; j++) {
            if (j < nb) {
                const float* xr = x + (size_t)(b0 + j) * cols + base;
                xl[j] = *reinterpret_cast<const float4*>(xr);
                xh[j] = *reinterpret_cast<const float4*>(xr + 32);
                sl[j] = xl[j].x + xl[j].y + xl[j].z + xl[j].w;
                sh[j] = xh[j].x + xh[j].y + xh[j].z + xh[j].w;
            }
        }
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            if (row0 + r < rows) {
                const unsigned char* b = w + (size_t)(row0 + r) * row_bytes + (size_t)sb * Q4K_BYTES;
                float d, dmin;
                q4k_dm(b, &d, &dmin);
                int sc1, m1, sc2, m2;
                q4k_scale_min(2 * j2, b + 4, &sc1, &m1);
                q4k_scale_min(2 * j2 + 1, b + 4, &sc2, &m2);
                int p = *(const int*)(b + 16 + 32 * j2 + 4 * sub);
                float a1 = d * sc1, c1 = dmin * m1, a2 = d * sc2, c2 = dmin * m2;
                float lo0 = p & 15, lo1 = (p >> 8) & 15, lo2 = (p >> 16) & 15, lo3 = (p >> 24) & 15;
                float hi0 = (p >> 4) & 15, hi1 = (p >> 12) & 15, hi2 = (p >> 20) & 15, hi3 = (p >> 28) & 15;
                #pragma unroll
                for (int j = 0; j < TBK; j++) {
                    if (j < nb) {
                        acc[r][j] += a1 * (lo0 * xl[j].x + lo1 * xl[j].y + lo2 * xl[j].z + lo3 * xl[j].w) - c1 * sl[j]
                                   + a2 * (hi0 * xh[j].x + hi1 * xh[j].y + hi2 * xh[j].z + hi3 * xh[j].w) - c2 * sh[j];
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++)
        #pragma unroll
        for (int j = 0; j < TBK; j++) {
            float v = warp_sum(acc[r][j]);
            if (lane == 0 && j < nb && row0 + r < rows) {
                size_t idx = (size_t)(b0 + j) * out_stride + out_off + row0 + r;
                out[idx] = accumulate ? out[idx] + v : v;
            }
        }
}

// Q6_K: 16 lanes per 128-value half; each lane takes 2 adjacent positions of all 4 groups (8 values).
template <int TBK>
__device__ void q6k_matvec(const unsigned char* __restrict__ w, const float* __restrict__ wd, const float* __restrict__ x,
                           float* __restrict__ out, int rows, int cols, int n, int accumulate, int out_stride, int out_off) {
    int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * RPW;
    int lane = threadIdx.x & 31;
    if (row0 >= rows) return;
    int b0 = blockIdx.y * TBK, nb = min(TBK, n - b0);
    int nsb = cols >> 8;
    int half = lane >> 4, l = 2 * (lane & 15), is = l >> 4;
    size_t row_bytes = (size_t)nsb * Q6K_BYTES;
    float acc[RPW][TBK];
    #pragma unroll
    for (int r = 0; r < RPW; r++)
        #pragma unroll
        for (int j = 0; j < TBK; j++) acc[r][j] = 0.f;
    #pragma unroll 4 // several super-blocks' loads in flight before any is used
    for (int sb = 0; sb < nsb; sb++) {
        float2 xv[4][TBK]; // [group][token]
        #pragma unroll
        for (int g = 0; g < 4; g++)
            #pragma unroll
            for (int j = 0; j < TBK; j++)
                if (j < nb) xv[g][j] = *reinterpret_cast<const float2*>(x + (size_t)(b0 + j) * cols + sb * 256 + 128 * half + 32 * g + l);
        #pragma unroll
        for (int r = 0; r < RPW; r++) {
            if (row0 + r < rows) {
                const unsigned char* b = w + (size_t)(row0 + r) * row_bytes + (size_t)sb * Q6K_BYTES;
                float d = wd[(size_t)(row0 + r) * nsb + sb];
                int qla =*(const unsigned short*)(b + 64 * half + l);
                int qlb = *(const unsigned short*)(b + 64 * half + 32 + l);
                int qh = *(const unsigned short*)(b + 128 + 32 * half + l);
                #pragma unroll
                for (int g = 0; g < 4; g++) {
                    int ql = (g & 1) ? qlb : qla;
                    int s4 = (g >= 2) ? 4 : 0;
                    float q0 = (float)((((ql >> s4) & 15) | (((qh >> (2 * g)) & 3) << 4)) - 32);
                    float q1 = (float)((((ql >> (8 + s4)) & 15) | (((qh >> (8 + 2 * g)) & 3) << 4)) - 32);
                    float s = d * (float)(signed char)b[192 + 8 * half + 2 * g + is];
                    #pragma unroll
                    for (int j = 0; j < TBK; j++)
                        if (j < nb) acc[r][j] += s * (q0 * xv[g][j].x + q1 * xv[g][j].y);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < RPW; r++)
        #pragma unroll
        for (int j = 0; j < TBK; j++) {
            float v = warp_sum(acc[r][j]);
            if (lane == 0 && j < nb && row0 + r < rows) {
                size_t idx = (size_t)(b0 + j) * out_stride + out_off + row0 + r;
                out[idx] = accumulate ? out[idx] + v : v;
            }
        }
}

// Decode matmuls with int8 activations (quantize_q8 / rmsnorm_q8 / silu_q8 first): weights stay packed
// and go straight into __dp4a (4 multiply-adds per instruction), instead of being unpacked to floats.
// Templated on NT, the most tokens per launch: NT = 1 for normal decoding, NT = 8 for speculative verify
// batches, where each weight block is read and decoded once and applied to every token (each token's
// activations come from cache as needed), and on R, the rows per warp. RPW rows per warp (K-quants).
//
// A token gets bit-identical results alone, in a batch of 8 or in a tensor-core verify batch (q4k_vfy /
// q6k_vfy), so speculative decoding reproduces plain greedy output exactly. Every K-quant decode kernel
// does the same arithmetic: integer sums over whole sub-blocks (exact, in any order), then the float steps
// below, spelled out with explicit roundings (no kernel's compiler may fuse them differently) and summed in
// one fixed order. A super-block's sub-blocks pair up (Q4_K: 2p and 2p + 1, p < 4; Q6_K: groups g and
// g + 2 of the same half and 16-value part, p = 4 half + 2 g + part < 8); each pair p has an accumulator per
// chain q of super-blocks (q, q + G, q + 2G, ...; Q4_K G = 4, Q6_K G = 2), and per super-block of the chain
//   s = sc_b dx_b I_b + sc_a dx_a I_a    (I: integer sum, dx: activation scale, sc: sub-block scale)
//   Q4_K  A = A + d s, then A = A - dmin m, with m = m_b sl_b + m_a sl_a (sl = dx * activation block sum)
//   Q6_K  A = A + d s
// and the row's result is, per chain, the pairs as a tree (((A0 + A1) + (A2 + A3)), Q6_K the same over 8),
// then the chains: Q4_K (T0 + T1) + (T2 + T3), Q6_K T0 + T1. The dp4a kernels' lanes are those pairs and
// chains, so all their cross-lane sums come at the end.
#define DP4A_PARAMS const unsigned char* __restrict__ w, const float* __restrict__ wd, const signed char* __restrict__ xq, \
                    const float* __restrict__ xd, const int* __restrict__ xs, float* __restrict__ out, int rows, int cols, int n, int accumulate, int out_stride, int out_off

__device__ __forceinline__ float pair_sf(float sca, float scb, float dxa, float dxb, float ia, float ib) {
    return __fmaf_rn(__fmul_rn(scb, dxb), ib, __fmul_rn(__fmul_rn(sca, dxa), ia));
}
__device__ __forceinline__ float pair_s(int sca, int scb, float dxa, float dxb, int ia, int ib) {
    return pair_sf((float)sca, (float)scb, dxa, dxb, (float)ia, (float)ib);
}
__device__ __forceinline__ float q4k_pair_mf(float ma, float mb, float sla, float slb) {
    return __fmaf_rn(mb, slb, __fmul_rn(ma, sla));
}
__device__ __forceinline__ float q4k_pair_m(int ma, int mb, float sla, float slb) {
    return q4k_pair_mf((float)ma, (float)mb, sla, slb);
}
__device__ __forceinline__ float q4k_step(float acc, float d, float dmin, float s, float m) {
    return __fmaf_rn(-dmin, m, __fmaf_rn(d, s, acc));
}
__device__ __forceinline__ float tree4(const float* s) {
    return __fadd_rn(__fadd_rn(s[0], s[1]), __fadd_rn(s[2], s[3]));
}

// Q4_K: each lane takes 16 bytes of a 32-byte nibble chunk = 16 values of each of two sub-blocks (32
// values), plus the block's 16-byte header (d, dmin and all 12 scale bytes) in one load; 8 lanes cover a
// super-block, so the warp walks four super-blocks per step. The R rows of a warp are decoded first, then
// each token's activations are loaded once for all of them. Lane pairs swap halves for whole sub-block
// sums and keep the same accumulator (the "min" term uses the activation block sums from the quantiser).
template <int NT, int R>
__device__ __forceinline__ void q4k_dp4a(DP4A_PARAMS) {
    int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R;
    int lane = threadIdx.x & 31;
    if (row0 >= rows) return;
    unsigned grp = 0xFFu << (lane & 24); // the 8 lanes on the same super-blocks (shuffles inside the loop)
    int nsb = cols >> 8;
    int qsel = lane >> 3, l8 = lane & 7, j2 = l8 >> 1, hf = l8 & 1;
    size_t row_bytes = (size_t)nsb * Q4K_BYTES;
    float acc[R][NT];
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int j = 0; j < NT; j++) acc[r][j] = 0.f;
    #pragma unroll 2 // more super-blocks' loads in flight before any is used
    for (int sb = qsel; sb < nsb; sb += 4) {
        int pos = sb * 256 + 64 * j2 + 16 * hf;
        int4 lo4[R], hi4[R];
        float d[R], dmin[R];
        int sc1[R], m1[R], sc2[R], m2[R];
        #pragma unroll
        for (int r = 0; r < R; r++) {
            int row = min(row0 + r, rows - 1); // past the end: computed, never stored
            const unsigned char* b = w + (size_t)row * row_bytes + (size_t)sb * Q4K_BYTES;
            int4 hdr = *reinterpret_cast<const int4*>(b);
            int4 p = *reinterpret_cast<const int4*>(b + 16 + 32 * j2 + 16 * hf);
            d[r] = h2f((unsigned)hdr.x & 0xFFFF);
            dmin[r] = h2f((unsigned)hdr.x >> 16);
            q4k_scale_min2w(2 * j2, hdr.y, hdr.z, hdr.w, &sc1[r], &m1[r], &sc2[r], &m2[r]);
            lo4[r] = make_int4(p.x & 0x0F0F0F0F, p.y & 0x0F0F0F0F, p.z & 0x0F0F0F0F, p.w & 0x0F0F0F0F);
            hi4[r] = make_int4((p.x >> 4) & 0x0F0F0F0F, (p.y >> 4) & 0x0F0F0F0F, (p.z >> 4) & 0x0F0F0F0F, (p.w >> 4) & 0x0F0F0F0F);
        }
        #pragma unroll
        for (int j = 0; j < NT; j++) {
            if (j < n) {
                const signed char* x = xq + (size_t)j * cols;
                const float* dx = xd + (size_t)j * (cols >> 5);
                const int* sx = xs + (size_t)j * (cols >> 5);
                int4 xl = *reinterpret_cast<const int4*>(x + pos), xh = *reinterpret_cast<const int4*>(x + pos + 32);
                float dl = dx[pos >> 5], dh = dx[(pos >> 5) + 1];
                float sl = __fmul_rn(dl, (float)sx[pos >> 5]), sh = __fmul_rn(dh, (float)sx[(pos >> 5) + 1]);
                #pragma unroll
                for (int r = 0; r < R; r++) {
                    int lo = __dp4a(lo4[r].x, xl.x, __dp4a(lo4[r].y, xl.y, __dp4a(lo4[r].z, xl.z, __dp4a(lo4[r].w, xl.w, 0))));
                    int hi = __dp4a(hi4[r].x, xh.x, __dp4a(hi4[r].y, xh.y, __dp4a(hi4[r].z, xh.z, __dp4a(hi4[r].w, xh.w, 0))));
                    lo += __shfl_xor_sync(grp, lo, 1); // the other 16 values of both sub-blocks
                    hi += __shfl_xor_sync(grp, hi, 1);
                    acc[r][j] = q4k_step(acc[r][j], d[r], dmin[r], pair_s(sc1[r], sc2[r], dl, dh, lo, hi), q4k_pair_m(m1[r], m2[r], sl, sh));
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int j = 0; j < NT; j++) {
            // the pairs (lane bits 1-2), then the chains (bits 3-4): ((A0 + A1) + (A2 + A3)) per chain, (T0 + T1) + (T2 + T3)
            float v = acc[r][j];
            #pragma unroll
            for (int o = 2; o < 32; o <<= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffff, v, o));
            if (lane == 0 && j < n && row0 + r < rows) {
                size_t idx = (size_t)j * out_stride + out_off + row0 + r;
                out[idx] = accumulate ? out[idx] + v : v;
            }
        }
}

// Q6_K: per 128-value half, 4 lanes take groups {0,2} and 4 lanes groups {1,3}; each lane 8 consecutive
// positions of both groups (they share the low-nibble bytes: low nibble = first group, high = second).
// A half-warp covers a super-block, so the warp walks two super-blocks per step. R rows decoded first.
// Lane pairs swap halves for whole 16-value sub-block sums and keep the same accumulator.
template <int NT, int R>
__device__ __forceinline__ void q6k_dp4a(DP4A_PARAMS) {
    int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R;
    int lane = threadIdx.x & 31;
    if (row0 >= rows) return;
    unsigned grp = 0xFFFFu << (lane & 16); // the 16 lanes on the same super-blocks
    int nsb = cols >> 8;
    int hl = lane >> 4, l16 = lane & 15;
    int half = l16 >> 3, gsel = (l16 >> 2) & 1, l = 8 * (l16 & 3), is = l >> 4;
    int ga = gsel, gb = gsel + 2;
    size_t row_bytes = (size_t)nsb * Q6K_BYTES;
    float acc[R][NT];
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int j = 0; j < NT; j++) acc[r][j] = 0.f;
    #pragma unroll 2 // more super-blocks' loads in flight before any is used
    for (int sb = hl; sb < nsb; sb += 2) {
        int pa = sb * 256 + 128 * half + 32 * ga + l, pb = pa + 64;
        int qa0[R], qa1[R], qb0[R], qb1[R], sa[R], sbb[R];
        float d[R];
        #pragma unroll
        for (int r = 0; r < R; r++) {
            int row = min(row0 + r, rows - 1); // past the end: computed, never stored
            const unsigned char* b = w + (size_t)row * row_bytes + (size_t)sb * Q6K_BYTES;
            d[r] = wd[(size_t)row * nsb + sb];
            int ql0 = *(const int*)(b + 64 * half + 32 * gsel + l), ql1 = *(const int*)(b + 64 * half + 32 * gsel + l + 4);
            int qh0 = *(const int*)(b + 128 + 32 * half + l), qh1 = *(const int*)(b + 132 + 32 * half + l);
            qa0[r] = __vsub4((ql0 & 0x0F0F0F0F) | (((qh0 >> (2 * ga)) & 0x03030303) << 4), 0x20202020);
            qa1[r] = __vsub4((ql1 & 0x0F0F0F0F) | (((qh1 >> (2 * ga)) & 0x03030303) << 4), 0x20202020);
            qb0[r] = __vsub4(((ql0 >> 4) & 0x0F0F0F0F) | (((qh0 >> (2 * gb)) & 0x03030303) << 4), 0x20202020);
            qb1[r] = __vsub4(((ql1 >> 4) & 0x0F0F0F0F) | (((qh1 >> (2 * gb)) & 0x03030303) << 4), 0x20202020);
            const signed char* sc = (const signed char*)(b + 192 + 8 * half + is);
            sa[r] = sc[2 * ga];
            sbb[r] = sc[2 * gb];
        }
        #pragma unroll
        for (int j = 0; j < NT; j++) {
            if (j < n) {
                const signed char* x = xq + (size_t)j * cols;
                const float* dx = xd + (size_t)j * (cols >> 5);
                int2 xa = *reinterpret_cast<const int2*>(x + pa), xb = *reinterpret_cast<const int2*>(x + pb);
                float da = dx[pa >> 5], db = dx[pb >> 5];
                #pragma unroll
                for (int r = 0; r < R; r++) {
                    int ia = __dp4a(qa0[r], xa.x, __dp4a(qa1[r], xa.y, 0)), ib = __dp4a(qb0[r], xb.x, __dp4a(qb1[r], xb.y, 0));
                    ia += __shfl_xor_sync(grp, ia, 1); // the other 8 values of both sub-blocks
                    ib += __shfl_xor_sync(grp, ib, 1);
                    acc[r][j] = __fmaf_rn(d[r], pair_s(sa[r], sbb[r], da, db, ia, ib), acc[r][j]);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int j = 0; j < NT; j++) {
            // the pairs (lane bits 1-3: part, group, half) as a tree, then the chains (bit 4)
            float v = acc[r][j];
            #pragma unroll
            for (int o = 2; o < 32; o <<= 1) v = __fadd_rn(v, __shfl_xor_sync(0xffffffff, v, o));
            if (lane == 0 && j < n && row0 + r < rows) {
                size_t idx = (size_t)j * out_stride + out_off + row0 + r;
                out[idx] = accumulate ? out[idx] + v : v;
            }
        }
}
// Q3_K, repacked at upload to 112-byte blocks: a 16-byte header (12 scale bytes, f16 d, 2 pad bytes),
// 32 high-bit bytes, 64 bytes of 2-bit pairs. Per 128-value half, 4 lanes take 8 consecutive positions
// each of all 4 groups (the groups share the bytes: group j is bits 2j of the pairs and bit 4*half + j of
// the high bits), so a lane decodes 32 values from two 8-byte loads. 8 lanes cover a super-block; the warp
// walks four super-blocks per step. Values are 2 low bits + 4 * high bit - 4, as signed bytes for dp4a.
#define Q3K_GPU_BYTES 112
// 6-bit scale i (0..15) from the 12 scale bytes as three words, centred to -32..31
__device__ __forceinline__ int q3k_scale(int i, unsigned w0, unsigned w1, unsigned w2) {
    unsigned lo = (((i & 7) < 4 ? w0 : w1) >> (8 * (i & 3) + (i < 8 ? 0 : 4))) & 0xF;
    unsigned hi = (w2 >> (8 * (i & 3) + 2 * (i >> 2))) & 3;
    return (int)(lo | (hi << 4)) - 32;
}
template <int NT, int R>
__device__ __forceinline__ void q3k_dp4a(DP4A_PARAMS) {
    int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * R;
    int lane = threadIdx.x & 31;
    if (row0 >= rows) return;
    int nsb = cols >> 8;
    int qsel = lane >> 3, l8 = lane & 7, half = l8 >> 2, l0 = 8 * (l8 & 3), sub = l0 >> 4;
    size_t row_bytes = (size_t)nsb * Q3K_GPU_BYTES;
    float acc[R][NT];
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int j = 0; j < NT; j++) acc[r][j] = 0.f;
    #pragma unroll 2 // more super-blocks' loads in flight before any is used
    for (int sb = qsel; sb < nsb; sb += 4) {
        int q[R][4][2], sc[R][4];
        float d[R];
        #pragma unroll
        for (int r = 0; r < R; r++) {
            int row = min(row0 + r, rows - 1); // past the end: computed, never stored
            const unsigned char* b = w + (size_t)row * row_bytes + (size_t)sb * Q3K_GPU_BYTES;
            int4 hdr = *reinterpret_cast<const int4*>(b);
            int2 hm = *reinterpret_cast<const int2*>(b + 16 + l0), qs = *reinterpret_cast<const int2*>(b + 48 + 32 * half + l0);
            d[r] = h2f((unsigned)hdr.w & 0xFFFF);
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                int hs = 4 * half + g; // shifts below are safe: every byte is masked
                q[r][g][0] = __vsub4(((qs.x >> (2 * g)) & 0x03030303) | (((hm.x >> hs) & 0x01010101) << 2), 0x04040404);
                q[r][g][1] = __vsub4(((qs.y >> (2 * g)) & 0x03030303) | (((hm.y >> hs) & 0x01010101) << 2), 0x04040404);
                sc[r][g] = q3k_scale(8 * half + 2 * g + sub, hdr.x, hdr.y, hdr.z);
            }
        }
        #pragma unroll
        for (int j = 0; j < NT; j++) {
            if (j < n) {
                const signed char* x = xq + (size_t)j * cols;
                const float* dx = xd + (size_t)j * (cols >> 5);
                #pragma unroll
                for (int g = 0; g < 4; g++) {
                    int pos = sb * 256 + 128 * half + 32 * g + l0;
                    int2 xv = *reinterpret_cast<const int2*>(x + pos);
                    float dg = dx[pos >> 5];
                    #pragma unroll
                    for (int r = 0; r < R; r++) acc[r][j] += d[r] * (float)sc[r][g] * dg * (float)__dp4a(q[r][g][0], xv.x, __dp4a(q[r][g][1], xv.y, 0));
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++)
        #pragma unroll
        for (int j = 0; j < NT; j++) {
            float v = warp_sum(acc[r][j]);
            if (lane == 0 && j < n && row0 + r < rows) {
                size_t idx = (size_t)j * out_stride + out_off + row0 + r;
                out[idx] = accumulate ? out[idx] + v : v;
            }
        }
}

// Q8_0 (used for draft models): one row per warp, each lane 16 int8 weights and 16 activations per step.
template <int NT>
__device__ __forceinline__ void q8_dp4a(DP4A_PARAMS) {
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const signed char* wr = reinterpret_cast<const signed char*>(w) + (size_t)row * cols;
    const float* dr = wd + (size_t)row * (cols >> 5);
    float acc[NT];
    #pragma unroll
    for (int j = 0; j < NT; j++) acc[j] = 0.f;
    #pragma unroll 2
    for (int i = lane * 16; i < cols; i += 512) {
        int4 wv = *reinterpret_cast<const int4*>(wr + i);
        float dw = dr[i >> 5];
        #pragma unroll
        for (int j = 0; j < NT; j++) {
            if (j < n) {
                int4 xv = *reinterpret_cast<const int4*>(xq + (size_t)j * cols + i);
                int s = __dp4a(wv.x, xv.x, __dp4a(wv.y, xv.y, __dp4a(wv.z, xv.z, __dp4a(wv.w, xv.w, 0))));
                acc[j] += dw * xd[(size_t)j * (cols >> 5) + (i >> 5)] * (float)s;
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < NT; j++) {
        float v = warp_sum(acc[j]);
        if (lane == 0 && j < n) {
            size_t idx = (size_t)j * out_stride + out_off + row;
            out[idx] = accumulate ? out[idx] + v : v;
        }
    }
}

// Verify batches (speculative decoding) on int8 tensor cores, sm_75 and newer: up to TILES x 8 tokens per
// launch, each given bit for bit what the dp4a kernels give it. An m8n8k16 product is an exact integer sum
// over 16 values (two chained, a 32-value sub-block); the float steps are theirs, in their order (above).
// A block of 4 warps owns 32 rows (4 tiles of 8) and TILES tiles of 8 tokens; per super-block it stages the
// rows' blocks and the tokens' activations (with their scales and scaled sums, computed once per token) in
// shared memory with coalesced 16-byte loads, padded so the fragment reads hit different banks, and fetches
// the next super-block's into registers while computing. Each warp takes one token tile and TILES row tiles
// (shared memory is the scarce part: a thread reads its two tokens' activations and scales once for all its
// rows); lane T holds the weights of row T/4 of a row tile and the activations of token T/4 of the token tile,
// and owns the results of row T/4 for tokens 2 (T%4) and 2 (T%4) + 1. The chains of super-blocks go one at a
// time, so a result needs only its pairs' accumulators and what the chain tree has summed so far.
__device__ __forceinline__ void mma_s8(int a, int b, int& c0, int& c1) {
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};" : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
}
#define VFY_ROWS 32
#define VFY_SC 12 // floats per staged token's 8 block scales (padded: bank spread)

// (float)i for |i| < 2^22, exactly, at full rate (the I2F instruction runs at a quarter of it)
__device__ __forceinline__ float i2f(int i) {
    return __fsub_rn(__int_as_float(0x4B400000 + i), 12582912.f);
}

// One super-block's staging, held in registers so the next one's loads are in flight while this one computes:
// the block's 32 rows (WN 16-byte pieces each, and Q6_K's d), the tokens' 256 activations, their 8 block scales
// and (SUMS) block sums. 128 threads.
template <int TILES, int WN, bool SUMS>
struct VfyStage {
    static constexpr int NW = (VFY_ROWS * WN + 127) / 128, NS = (TILES + 1) / 2;
    int4 w[NW], x[TILES];
    float dx[NS], d;
    int sx[NS];
    __device__ __forceinline__ void fetch(const unsigned char* wq, const float* wd, int bytes, int rows, int row0, int nsb,
                                          const signed char* xq, const float* xd, const int* xs, int n, int cols, int sb) {
        int tid = threadIdx.x, ntok = ((n + 7) >> 3) * 8, nb = cols >> 5;
        #pragma unroll
        for (int k = 0; k < NW; k++) {
            int i = tid + 128 * k;
            if (i < VFY_ROWS * WN) {
                int rr = i / WN, c = i - WN * rr;
                w[k] = *reinterpret_cast<const int4*>(wq + ((size_t)min(row0 + rr, rows - 1) * nsb + sb) * bytes + 16 * c);
            }
        }
        if (wd && tid < VFY_ROWS) d = wd[(size_t)min(row0 + tid, rows - 1) * nsb + sb];
        #pragma unroll
        for (int k = 0; k < TILES; k++) {
            int i = tid + 128 * k;
            if (i < ntok * 16) x[k] = *reinterpret_cast<const int4*>(xq + (size_t)min(i >> 4, n - 1) * cols + sb * 256 + 16 * (i & 15));
        }
        #pragma unroll
        for (int k = 0; k < NS; k++) {
            int i = tid + 128 * k;
            if (i < ntok * 8) {
                size_t at = (size_t)min(i >> 3, n - 1) * nb + sb * 8 + (i & 7);
                dx[k] = xd[at];
                if (SUMS) sx[k] = xs[at];
            }
        }
    }
    __device__ __forceinline__ void store(int* ws, int vw, float* sd, int* xsh, int vx, float* sdx, float* ssl, int n) {
        int tid = threadIdx.x, ntok = ((n + 7) >> 3) * 8;
        #pragma unroll
        for (int k = 0; k < NW; k++) {
            int i = tid + 128 * k;
            if (i < VFY_ROWS * WN) {
                int rr = i / WN;
                *reinterpret_cast<int4*>(ws + rr * vw + 4 * (i - WN * rr)) = w[k];
            }
        }
        if (sd && tid < VFY_ROWS) sd[tid] = d;
        #pragma unroll
        for (int k = 0; k < TILES; k++) {
            int i = tid + 128 * k;
            if (i < ntok * 16) *reinterpret_cast<int4*>(xsh + (i >> 4) * vx + 4 * (i & 15)) = x[k];
        }
        #pragma unroll
        for (int k = 0; k < NS; k++) {
            int i = tid + 128 * k;
            if (i < ntok * 8) {
                sdx[(i >> 3) * VFY_SC + (i & 7)] = dx[k];
                if (SUMS) ssl[(i >> 3) * VFY_SC + (i & 7)] = __fmul_rn(dx[k], (float)sx[k]);
            }
        }
    }
};

// The super-blocks of chain q (q, q + G, q + 2G, ...) in order, chain after chain: the step after sb.
__device__ __forceinline__ bool vfy_next(int G, int nsb, int q, int sb, int* nq, int* nsb2) {
    *nq = q;
    *nsb2 = sb + G;
    if (*nsb2 >= nsb) {
        *nq = q + 1;
        *nsb2 = q + 1;
    }
    return *nq < G && *nsb2 < nsb;
}

#define Q4K_VW 40 // words per staged Q4_K row block (36 + padding)
#define Q4K_VX 72 // words per staged token (64 + padding)
template <int TILES>
__device__ __forceinline__ void q4k_vfy(DP4A_PARAMS) {
    __shared__ __align__(16) int ws[VFY_ROWS * Q4K_VW];
    __shared__ __align__(16) int xsh[TILES * 8 * Q4K_VX];
    __shared__ __align__(16) float sdx[TILES * 8 * VFY_SC], ssl[TILES * 8 * VFY_SC];
    int tid = threadIdx.x, lane = tid & 31, tq = lane & 3, g = lane >> 2, wp = tid >> 5;
    int tt = wp / (4 / TILES), rt0 = (wp % (4 / TILES)) * TILES; // this warp's token tile and first row tile
    int row0 = blockIdx.x * VFY_ROWS;
    if (row0 >= rows) return;
    int ntiles = (n + 7) >> 3, nsb = cols >> 8;
    VfyStage<TILES, 9, true> st;
    st.fetch(w, nullptr, Q4K_BYTES, rows, row0, nsb, xq, xd, xs, n, cols, 0);
    float acc[TILES][2][4], held[TILES][2], left[TILES][2]; // [row tile][result token][pair]
    #pragma unroll
    for (int k = 0; k < TILES; k++)
        #pragma unroll
        for (int u = 0; u < 2; u++)
            #pragma unroll
            for (int p = 0; p < 4; p++) acc[k][u][p] = 0.f;
    for (int q = 0, sb = 0;;) {
        __syncthreads(); // the last super-block's reads are done
        st.store(ws, Q4K_VW, nullptr, xsh, Q4K_VX, sdx, ssl, n);
        __syncthreads();
        int nq, nsb2;
        bool more = vfy_next(4, nsb, q, sb, &nq, &nsb2);
        if (more) st.fetch(w, nullptr, Q4K_BYTES, rows, row0, nsb, xq, xd, xs, n, cols, nsb2);
        if (tt < ntiles) {
            // this thread's token of the tile (B fragments: bytes 64 p + 8 tq .. + 7, and + 32), its two results' scales
            const int* xr = xsh + (8 * tt + g) * Q4K_VX + 2 * tq;
            int2 xa[4], xb[4];
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                xa[p] = *reinterpret_cast<const int2*>(xr + 16 * p);
                xb[p] = *reinterpret_cast<const int2*>(xr + 16 * p + 8);
            }
            float dx[2][8], sl[2][8];
            #pragma unroll
            for (int u = 0; u < 2; u++) {
                const float* dr = sdx + (8 * tt + 2 * tq + u) * VFY_SC;
                const float* sr = ssl + (8 * tt + 2 * tq + u) * VFY_SC;
                float4 d0 = *reinterpret_cast<const float4*>(dr), d1 = *reinterpret_cast<const float4*>(dr + 4);
                float4 s0 = *reinterpret_cast<const float4*>(sr), s1 = *reinterpret_cast<const float4*>(sr + 4);
                dx[u][0] = d0.x; dx[u][1] = d0.y; dx[u][2] = d0.z; dx[u][3] = d0.w; dx[u][4] = d1.x; dx[u][5] = d1.y; dx[u][6] = d1.z; dx[u][7] = d1.w;
                sl[u][0] = s0.x; sl[u][1] = s0.y; sl[u][2] = s0.z; sl[u][3] = s0.w; sl[u][4] = s1.x; sl[u][5] = s1.y; sl[u][6] = s1.z; sl[u][7] = s1.w;
            }
            #pragma unroll
            for (int k = 0; k < TILES; k++) {
                const int* wr = ws + (8 * (rt0 + k) + g) * Q4K_VW;
                int4 hdr = *reinterpret_cast<const int4*>(wr);
                float d = h2f((unsigned)hdr.x & 0xFFFF), dmin = h2f((unsigned)hdr.x >> 16);
                #pragma unroll
                for (int p = 0; p < 4; p++) {
                    int s1, m1, s2, m2;
                    q4k_scale_min2w(2 * p, hdr.y, hdr.z, hdr.w, &s1, &m1, &s2, &m2);
                    float sca = i2f(s1), ma = i2f(m1), scb = i2f(s2), mb = i2f(m2);
                    // nibble bytes 32 p + 8 tq .. + 7: values 8 tq .. + 7 of sub-blocks 2p (low) and 2p + 1 (high)
                    int2 wq = *reinterpret_cast<const int2*>(wr + 4 + 8 * p + 2 * tq);
                    int a0 = 0, a1 = 0, b0 = 0, b1 = 0;
                    mma_s8(wq.x & 0x0F0F0F0F, xa[p].x, a0, a1);
                    mma_s8(wq.y & 0x0F0F0F0F, xa[p].y, a0, a1);
                    mma_s8((wq.x >> 4) & 0x0F0F0F0F, xb[p].x, b0, b1);
                    mma_s8((wq.y >> 4) & 0x0F0F0F0F, xb[p].y, b0, b1);
                    int a = 2 * p, b = a + 1;
                    acc[k][0][p] = q4k_step(acc[k][0][p], d, dmin, pair_sf(sca, scb, dx[0][a], dx[0][b], i2f(a0), i2f(b0)), q4k_pair_mf(ma, mb, sl[0][a], sl[0][b]));
                    acc[k][1][p] = q4k_step(acc[k][1][p], d, dmin, pair_sf(sca, scb, dx[1][a], dx[1][b], i2f(a1), i2f(b1)), q4k_pair_mf(ma, mb, sl[1][a], sl[1][b]));
                }
            }
        }
        if (!more || nq != q) {
            // chain q is done (and, at the end, any empty ones after it): its pairs as a tree, then (T0 + T1) + (T2 + T3)
            for (int c = q; c < (more ? nq : 4); c++) {
                #pragma unroll
                for (int k = 0; k < TILES; k++)
                    #pragma unroll
                    for (int u = 0; u < 2; u++) {
                        float tc = c == q ? tree4(acc[k][u]) : 0.f;
                        if (c == 0 || c == 2) held[k][u] = tc;
                        else if (c == 1) left[k][u] = __fadd_rn(held[k][u], tc);
                        else held[k][u] = __fadd_rn(left[k][u], __fadd_rn(held[k][u], tc));
                        #pragma unroll
                        for (int p = 0; p < 4; p++) acc[k][u][p] = 0.f;
                    }
            }
        }
        if (!more) break;
        q = nq;
        sb = nsb2;
    }
    if (tt >= ntiles) return;
    #pragma unroll
    for (int k = 0; k < TILES; k++)
        #pragma unroll
        for (int u = 0; u < 2; u++) {
            int tok = 8 * tt + 2 * tq + u, row = row0 + 8 * (rt0 + k) + g;
            if (tok < n && row < rows) {
                size_t idx = (size_t)tok * out_stride + out_off + row;
                out[idx] = accumulate ? out[idx] + held[k][u] : held[k][u];
            }
        }
}

#define Q6K_VW 68 // words per staged Q6_K row block (52 + padding)
#define Q6K_VX 68 // words per staged token (64 + padding)
template <int TILES>
__device__ __forceinline__ void q6k_vfy(DP4A_PARAMS) {
    __shared__ __align__(16) int ws[VFY_ROWS * Q6K_VW];
    __shared__ __align__(16) int xsh[TILES * 8 * Q6K_VX];
    __shared__ __align__(16) float sdx[TILES * 8 * VFY_SC];
    __shared__ float sd[VFY_ROWS];
    int tid = threadIdx.x, lane = tid & 31, tq = lane & 3, g = lane >> 2, wp = tid >> 5;
    int tt = wp / (4 / TILES), rt0 = (wp % (4 / TILES)) * TILES;
    int row0 = blockIdx.x * VFY_ROWS;
    if (row0 >= rows) return;
    int ntiles = (n + 7) >> 3, nsb = cols >> 8;
    VfyStage<TILES, 13, false> st;
    st.fetch(w, wd, Q6K_BYTES, rows, row0, nsb, xq, xd, xs, n, cols, 0);
    float acc[TILES][2][8], held[TILES][2];
    #pragma unroll
    for (int k = 0; k < TILES; k++)
        #pragma unroll
        for (int u = 0; u < 2; u++)
            #pragma unroll
            for (int p = 0; p < 8; p++) acc[k][u][p] = 0.f;
    for (int q = 0, sb = 0;;) {
        __syncthreads();
        st.store(ws, Q6K_VW, sd, xsh, Q6K_VX, sdx, nullptr, n);
        __syncthreads();
        int nq, nsb2;
        bool more = vfy_next(2, nsb, q, sb, &nq, &nsb2);
        if (more) st.fetch(w, wd, Q6K_BYTES, rows, row0, nsb, xq, xd, xs, n, cols, nsb2);
        if (tt < ntiles) {
            const int* xr = xsh + (8 * tt + g) * Q6K_VX + tq;
            int xv[2][4][2]; // B fragments of sub-block (half, group, part)
            #pragma unroll
            for (int h = 0; h < 2; h++)
                #pragma unroll
                for (int gr = 0; gr < 4; gr++)
                    #pragma unroll
                    for (int is = 0; is < 2; is++) xv[h][gr][is] = xr[32 * h + 8 * gr + 4 * is];
            float dx[2][8];
            #pragma unroll
            for (int u = 0; u < 2; u++) {
                const float* dr = sdx + (8 * tt + 2 * tq + u) * VFY_SC;
                float4 d0 = *reinterpret_cast<const float4*>(dr), d1 = *reinterpret_cast<const float4*>(dr + 4);
                dx[u][0] = d0.x; dx[u][1] = d0.y; dx[u][2] = d0.z; dx[u][3] = d0.w; dx[u][4] = d1.x; dx[u][5] = d1.y; dx[u][6] = d1.z; dx[u][7] = d1.w;
            }
            #pragma unroll
            for (int k = 0; k < TILES; k++) {
                int rr = 8 * (rt0 + k) + g;
                const int* wr = ws + rr * Q6K_VW;
                float d = sd[rr];
                int4 scv = *reinterpret_cast<const int4*>(wr + 48);
                int words[4] = {scv.x, scv.y, scv.z, scv.w};
                #pragma unroll
                for (int h = 0; h < 2; h++)
                    #pragma unroll
                    for (int is = 0; is < 2; is++) {
                        // values 4 tq .. + 3 of sub-blocks (h, group, is): low nibbles for groups 0-1, high for 2-3
                        int qh = wr[32 + 8 * h + 4 * is + tq];
                        #pragma unroll
                        for (int gs = 0; gs < 2; gs++) {
                            int ql = wr[16 * h + 8 * gs + 4 * is + tq];
                            int wa = __vsub4((ql & 0x0F0F0F0F) | (((qh >> (2 * gs)) & 0x03030303) << 4), 0x20202020);
                            int wb = __vsub4(((ql >> 4) & 0x0F0F0F0F) | (((qh >> (2 * gs + 4)) & 0x03030303) << 4), 0x20202020);
                            int a0 = 0, a1 = 0, b0 = 0, b1 = 0;
                            mma_s8(wa, xv[h][gs][is], a0, a1);
                            mma_s8(wb, xv[h][gs + 2][is], b0, b1);
                            int pp = 4 * h + 2 * gs + is, sa = 8 * h + 2 * gs + is;
                            float sca = i2f((int)(signed char)((words[sa >> 2] >> (8 * (sa & 3))) & 0xFF));
                            float scb = i2f((int)(signed char)((words[(sa + 4) >> 2] >> (8 * ((sa + 4) & 3))) & 0xFF));
                            acc[k][0][pp] = __fmaf_rn(d, pair_sf(sca, scb, dx[0][4 * h + gs], dx[0][4 * h + gs + 2], i2f(a0), i2f(b0)), acc[k][0][pp]);
                            acc[k][1][pp] = __fmaf_rn(d, pair_sf(sca, scb, dx[1][4 * h + gs], dx[1][4 * h + gs + 2], i2f(a1), i2f(b1)), acc[k][1][pp]);
                        }
                    }
            }
        }
        if (!more || nq != q) {
            for (int c = q; c < (more ? nq : 2); c++) {
                #pragma unroll
                for (int k = 0; k < TILES; k++)
                    #pragma unroll
                    for (int u = 0; u < 2; u++) {
                        float tc = c == q ? __fadd_rn(tree4(acc[k][u]), tree4(acc[k][u] + 4)) : 0.f;
                        held[k][u] = c == 0 ? tc : __fadd_rn(held[k][u], tc);
                        #pragma unroll
                        for (int p = 0; p < 8; p++) acc[k][u][p] = 0.f;
                    }
            }
        }
        if (!more) break;
        q = nq;
        sb = nsb2;
    }
    if (tt >= ntiles) return;
    #pragma unroll
    for (int k = 0; k < TILES; k++)
        #pragma unroll
        for (int u = 0; u < 2; u++) {
            int tok = 8 * tt + 2 * tq + u, row = row0 + 8 * (rt0 + k) + g;
            if (tok < n && row < rows) {
                size_t idx = (size_t)tok * out_stride + out_off + row;
                out[idx] = accumulate ? out[idx] + held[k][u] : held[k][u];
            }
        }
}

extern "C" {

// the batch instances get a register budget, so the compiler cannot hoist every token's loads at once
#define DP4A_ARGS w, wd, xq, xd, xs, out, rows, cols, n, accumulate, out_stride, out_off
__global__ void matmul_q4k_dp4a(DP4A_PARAMS) { q4k_dp4a<1, RPW>(DP4A_ARGS); }
__global__ void __launch_bounds__(256, DP4A_BATCH_MINB) matmul_q4k_dp4a8(DP4A_PARAMS) { q4k_dp4a<8, DP4A_BATCH_RPW>(DP4A_ARGS); }
__global__ void matmul_q6k_dp4a(DP4A_PARAMS) { q6k_dp4a<1, RPW>(DP4A_ARGS); }
__global__ void __launch_bounds__(256, DP4A_BATCH_MINB) matmul_q6k_dp4a8(DP4A_PARAMS) { q6k_dp4a<8, DP4A_BATCH_RPW>(DP4A_ARGS); }
__global__ void matmul_q8_dp4a(DP4A_PARAMS) { q8_dp4a<1>(DP4A_ARGS); }
__global__ void __launch_bounds__(256, DP4A_BATCH_MINB) matmul_q8_dp4a8(DP4A_PARAMS) { q8_dp4a<8>(DP4A_ARGS); }
__global__ void matmul_q3k_dp4a(DP4A_PARAMS) { q3k_dp4a<1, RPW>(DP4A_ARGS); }
__global__ void __launch_bounds__(256, DP4A_BATCH_MINB) matmul_q3k_dp4a8(DP4A_PARAMS) { q3k_dp4a<8, DP4A_BATCH_RPW>(DP4A_ARGS); }
// verify batches of up to 8, 16 and 32 tokens (128 threads: 4 warps of 8 rows)
__global__ void __launch_bounds__(128) matmul_q4k_vfy1(DP4A_PARAMS) { q4k_vfy<1>(DP4A_ARGS); }
__global__ void __launch_bounds__(128) matmul_q4k_vfy2(DP4A_PARAMS) { q4k_vfy<2>(DP4A_ARGS); }
__global__ void __launch_bounds__(128) matmul_q4k_vfy4(DP4A_PARAMS) { q4k_vfy<4>(DP4A_ARGS); }
__global__ void __launch_bounds__(128) matmul_q6k_vfy1(DP4A_PARAMS) { q6k_vfy<1>(DP4A_ARGS); }
__global__ void __launch_bounds__(128) matmul_q6k_vfy2(DP4A_PARAMS) { q6k_vfy<2>(DP4A_ARGS); }
__global__ void __launch_bounds__(128) matmul_q6k_vfy4(DP4A_PARAMS) { q6k_vfy<4>(DP4A_ARGS); }
// Mixture of experts, one token, for the experts held in GPU memory. ptrs is a table of matrix
// addresses: gate and up of expert k at 2k and 2k + 1, its down at 2 * nk + k; blockIdx.y picks the entry.
// gate / up: row block y of out gets matrix y applied to the token's int8 activations.
__global__ void moe_gate_up_q3k(const unsigned long long* __restrict__ ptrs, const signed char* __restrict__ xq, const float* __restrict__ xd,
                                const int* __restrict__ xs, float* __restrict__ out, int rows, int cols) {
    q3k_dp4a<1, 1>(reinterpret_cast<const unsigned char*>(ptrs[blockIdx.y]), nullptr, xq, xd, xs, out + (size_t)blockIdx.y * rows, rows, cols, 1, 0, 0, 0);
}
// down: expert k reads its own activations (row k of xq / xd / xs, written by silu_q8)
__global__ void moe_down_q4k(const unsigned long long* __restrict__ ptrs, int nk, const signed char* __restrict__ xq, const float* __restrict__ xd,
                             const int* __restrict__ xs, float* __restrict__ out, int rows, int cols) {
    int k = blockIdx.y;
    q4k_dp4a<1, 1>(reinterpret_cast<const unsigned char*>(ptrs[2 * nk + k]), nullptr, xq + (size_t)k * cols, xd + (size_t)k * (cols >> 5),
                   xs + (size_t)k * (cols >> 5), out + (size_t)k * rows, rows, cols, 1, 0, 0, 0);
}
// x += cpu (the experts computed on the CPU; may be null) + sum over k of wts[k] * outs[k]
__global__ void moe_combine(float* __restrict__ x, const float* __restrict__ outs, const float* __restrict__ wts, int nk,
                            const float* __restrict__ cpu, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    float v = cpu ? cpu[i] : 0.f;
    for (int k = 0; k < nk; k++) v += wts[k] * outs[(size_t)k * dim + i];
    x[i] += v;
}

#define KQ_ARGS const unsigned char* __restrict__ w, const float* __restrict__ wd, const float* __restrict__ x, \
                float* __restrict__ out, int rows, int cols, int n, int accumulate, int out_stride, int out_off
__global__ void matmul_q4k_1(KQ_ARGS) { q4k_matvec<1>(w, x, out, rows, cols, n, accumulate, out_stride, out_off); }
__global__ void matmul_q4k_4(KQ_ARGS) { q4k_matvec<4>(w, x, out, rows, cols, n, accumulate, out_stride, out_off); }
__global__ void matmul_q6k_1(KQ_ARGS) { q6k_matvec<1>(w, wd, x, out, rows, cols, n, accumulate, out_stride, out_off); }
__global__ void matmul_q6k_4(KQ_ARGS) { q6k_matvec<4>(w, wd, x, out, rows, cols, n, accumulate, out_stride, out_off); }

#define ATT_MAX_CHUNK 256
#ifndef ATT_CHUNK_CAP
#define ATT_CHUNK_CAP ATT_MAX_CHUNK // set from the host (att_chunk_cap)
#endif
#define ATT_MAX_HD 128
#define ATT_MAX_G 8
#define ATT_SPLIT_MIN_SPAN 32

// Context slice per decode-attention block for a token that sees span positions: whole contexts up to
// ATT_SPLIT_MIN_SPAN, otherwise about want slices of 32..ATT_CHUNK_CAP positions. It depends on the
// token's own span only, so a token is attended the same way alone or inside a batch.
__device__ __forceinline__ int att_chunk_for(int span, int want) {
    return span <= ATT_SPLIT_MIN_SPAN ? ATT_MAX_CHUNK : min(max((span + want - 1) / want, 32), ATT_CHUNK_CAP);
}

// A lane's share of a decode attention score: per (<= 4) floats of the query and the key, in this order.
__device__ __forceinline__ float lane_dot(const float* q, const float* k) {
    return __fmaf_rn(q[3], k[3], __fmaf_rn(q[2], k[2], __fmaf_rn(q[1], k[1], __fmul_rn(q[0], k[0]))));
}

// Group l (4 values) of a key row: an int8 cache's row is read whole into registers first (kw: 32 words,
// eight 16-byte loads), an f16 one group by group from memory.
#ifdef KV8
__device__ __forceinline__ float4 att_kg(const int* kw, const kv_t*, int l) {
    int w = kw[l];
    return make_float4((float)(signed char)w, (float)(signed char)(w >> 8), (float)(signed char)(w >> 16), (float)(w >> 24));
}
#else
__device__ __forceinline__ float4 att_kg(const int*, const kv_t* kt, int l) { return kv_load4(kt + 4 * l); }
#endif

// S1_j of a key row for every head: lane j's and lane j + 16's products of attention_partial_gqa's warp
// version (groups j and j + 16 of the row, head size 128), summed as its first butterfly step sums them.
__device__ __forceinline__ void att_s1(const int* kw, const kv_t* kt, const float (*qsh)[ATT_MAX_HD], int G, int j, float* s) {
    float4 ka = att_kg(kw, kt, j), kb = att_kg(kw, kt, j + 16);
    float a4[4] = {ka.x, ka.y, ka.z, ka.w}, b4[4] = {kb.x, kb.y, kb.z, kb.w};
    #pragma unroll
    for (int g = 0; g < ATT_MAX_G; g++) {
        if (g < G) {
            float4 qa = *reinterpret_cast<const float4*>(&qsh[g][4 * j]), qc = *reinterpret_cast<const float4*>(&qsh[g][4 * j + 64]);
            float qa4[4] = {qa.x, qa.y, qa.z, qa.w}, qc4[4] = {qc.x, qc.y, qc.z, qc.w};
            s[g] = __fadd_rn(lane_dot(qa4, a4), lane_dot(qc4, b4));
        }
    }
}

// Flash-decoding attention for grouped-query models, part 1: one block per (kv head, token, slice of
// `chunk` positions) computes all G = n_heads / n_kv query heads that share the kv head, so each key
// and value row is read once. A warp scores one position at a time: every lane loads hd/32 consecutive
// floats of the key (one coalesced row read) and the dot products are reduced across the warp. The float
// steps are spelled out: attention_vfy gives a token in a verify batch exactly these results.
// Writes each head's unnormalised output, max score and exp-sum to `part`, or (nsplit == 1) the final
// normalised result to `out`. Needs hd a multiple of 32 (<= 128) or 256, and G <= ATT_MAX_G.
__global__ void attention_partial_gqa(const float* __restrict__ qkv, const kv_t* __restrict__ kc, const kv_t* __restrict__ vc,
                                      const float* __restrict__ ks, const float* __restrict__ vs,
                                      float* __restrict__ part, float* __restrict__ out, int pos0, const int* __restrict__ posp,
                                      int n_heads, int n_kv, int hd, float scale, int nsplit, int want) {
    if (posp) pos0 = *posp;
    __shared__ __align__(16) float sc[ATT_MAX_CHUNK][ATT_MAX_G]; // per position, its G heads together
    __shared__ __align__(16) float qsh[ATT_MAX_G][ATT_MAX_HD];   // the scaled queries (head size 128)
    __shared__ float red[32];
    int kvh = blockIdx.x, b = blockIdx.y, s = blockIdx.z;
    int G = n_heads / n_kv;
    int span = pos0 + b + 1; // causal
    int chunk = att_chunk_for(span, want);
    int t0 = s * chunk, len = min(span, t0 + chunk) - t0;
    int q_dim = n_heads * hd, kv_dim = n_kv * hd;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nwarps = blockDim.x >> 5;
    if (len <= 0) { // same for the whole block, so returning before the barriers is safe
        for (int g = 0; g < G; g++) {
            float* p = part + (((size_t)b * n_heads + kvh * G + g) * nsplit + s) * (hd + 2);
            for (int d = tid; d < hd; d += blockDim.x) p[d] = 0.f;
            if (tid == 0) { p[hd] = NEG_INF; p[hd + 1] = 0.f; }
        }
        return;
    }
    if (hd == 128) {
        // scores, a thread per position. The warp version below gives lane l the dot product of elements
        // 4l..4l+3 and sums the 32 by an xor butterfly (offsets 16, 8, 4, 2, 1): every lane ends with
        // ((S1_0 + S1_8) + (S1_4 + S1_12)) + ... where S1_j = d_j + d_(j+16). The same tree here, from the
        // same products in the same order, gives each score bit for bit; no shuffles, and a warp scores 32
        // positions at once instead of one.
        const float* qb = qkv + (size_t)b * (q_dim + 2 * kv_dim) + kvh * G * 128;
        for (int i = tid; i < G * 128; i += blockDim.x) qsh[i >> 7][i & 127] = __fmul_rn(qb[i], scale);
        __syncthreads();
        for (int t = tid; t < len; t += blockDim.x) {
            // the queries are read from shared memory as they are needed: hoisted out of the loop, all of them
            // would take far more registers than there are
            asm volatile("" ::: "memory");
            const kv_t* kt = kc + (size_t)(t0 + t) * kv_dim + kvh * 128;
#ifdef KV8
            // the row in eight 16-byte loads: a thread reads its own row, so narrower loads would touch a
            // warp's 32 rows (cache lines) once per 4 bytes
            int kw[32];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int4 v = *reinterpret_cast<const int4*>(kt + 16 * i);
                kw[4 * i] = v.x, kw[4 * i + 1] = v.y, kw[4 * i + 2] = v.z, kw[4 * i + 3] = v.w;
            }
#else
            int kw[1] = {0};
#endif
            // the butterfly's tree: S2_j = S1_j + S1_(j+8), S3_j = S2_j + S2_(j+4), S4_j = S3_j + S3_(j+2),
            // score = S4_0 + S4_1, walked in bit-reversed j so each level keeps one pending sum per head
            float s[ATT_MAX_G], l1[ATT_MAX_G], l2[ATT_MAX_G], l3[ATT_MAX_G], l4[ATT_MAX_G];
            #define ATT_PUT(lv) _Pragma("unroll") for (int g = 0; g < ATT_MAX_G; g++) if (g < G) lv[g] = s[g];
            #define ATT_UP(lv, from) _Pragma("unroll") for (int g = 0; g < ATT_MAX_G; g++) if (g < G) lv[g] = __fadd_rn(from[g], s[g]);
            #define ATT_FOLD(dst, lv) _Pragma("unroll") for (int g = 0; g < ATT_MAX_G; g++) if (g < G) s[g] = __fadd_rn(lv[g], s[g]);
            att_s1(kw, kt, qsh, G, 0, s);  ATT_PUT(l1)
            att_s1(kw, kt, qsh, G, 8, s);  ATT_UP(l2, l1)                                   // S2_0
            att_s1(kw, kt, qsh, G, 4, s);  ATT_PUT(l1)
            att_s1(kw, kt, qsh, G, 12, s); ATT_FOLD(s, l1) ATT_UP(l3, l2)                   // S2_4, S3_0
            att_s1(kw, kt, qsh, G, 2, s);  ATT_PUT(l1)
            att_s1(kw, kt, qsh, G, 10, s); ATT_UP(l2, l1)                                   // S2_2
            att_s1(kw, kt, qsh, G, 6, s);  ATT_PUT(l1)
            att_s1(kw, kt, qsh, G, 14, s); ATT_FOLD(s, l1) ATT_FOLD(s, l2) ATT_UP(l4, l3)   // S2_6, S3_2, S4_0
            att_s1(kw, kt, qsh, G, 1, s);  ATT_PUT(l1)
            att_s1(kw, kt, qsh, G, 9, s);  ATT_UP(l2, l1)                                   // S2_1
            att_s1(kw, kt, qsh, G, 5, s);  ATT_PUT(l1)
            att_s1(kw, kt, qsh, G, 13, s); ATT_FOLD(s, l1) ATT_UP(l3, l2)                   // S2_5, S3_1
            att_s1(kw, kt, qsh, G, 3, s);  ATT_PUT(l1)
            att_s1(kw, kt, qsh, G, 11, s); ATT_UP(l2, l1)                                   // S2_3
            att_s1(kw, kt, qsh, G, 7, s);  ATT_PUT(l1)
            att_s1(kw, kt, qsh, G, 15, s); ATT_FOLD(s, l1) ATT_FOLD(s, l2) ATT_FOLD(s, l3) ATT_FOLD(s, l4) // S2_7 .. score
            #undef ATT_PUT
            #undef ATT_UP
            #undef ATT_FOLD
            float kscale = KV_SCALE(ks, t0 + t, n_kv, kvh);
            #pragma unroll
            for (int g = 0; g < ATT_MAX_G; g++) if (g < G) sc[t][g] = __fmul_rn(s[g], kscale);
        }
    } else if (hd == 256) {
    // wide heads: a lane holds 8 consecutive floats of the query and the key, as two groups of 4
    float q[ATT_MAX_G][8];
    const float* qrow = qkv + (size_t)b * (q_dim + 2 * kv_dim) + kvh * G * hd + lane * 8;
    #pragma unroll
    for (int g = 0; g < ATT_MAX_G; g++)
        #pragma unroll
        for (int e = 0; e < 8; e++) q[g][e] = g < G ? __fmul_rn(qrow[g * hd + e], scale) : 0.f;
    for (int t = warp; t < len; t += nwarps) {
        const kv_t* kt = kc + (size_t)(t0 + t) * kv_dim + kvh * hd + lane * 8;
        float4 v0 = kv_load4(kt), v1 = kv_load4(kt + 4);
        float k0[4] = {v0.x, v0.y, v0.z, v0.w}, k1[4] = {v1.x, v1.y, v1.z, v1.w};
        float kscale = KV_SCALE(ks, t0 + t, n_kv, kvh);
        #pragma unroll
        for (int g = 0; g < ATT_MAX_G; g++) {
            if (g < G) {
                float d = __fadd_rn(lane_dot(q[g], k0), lane_dot(q[g] + 4, k1));
                for (int o = 16; o > 0; o >>= 1) d = __fadd_rn(d, __shfl_xor_sync(0xffffffff, d, o));
                if (lane == 0) sc[t][g] = __fmul_rn(d, kscale);
            }
        }
    }
    } else {
    int per = hd >> 5; // floats per lane: 2 (hd 64) or 4 (hd 128)
    float q[ATT_MAX_G][4];
    const float* qrow = qkv + (size_t)b * (q_dim + 2 * kv_dim) + kvh * G * hd + lane * per;
    #pragma unroll
    for (int g = 0; g < ATT_MAX_G; g++)
        #pragma unroll
        for (int e = 0; e < 4; e++) q[g][e] = (g < G && e < per) ? __fmul_rn(qrow[g * hd + e], scale) : 0.f;

    for (int t = warp; t < len; t += nwarps) {
        const kv_t* kt = kc + (size_t)(t0 + t) * kv_dim + kvh * hd + lane * per;
        float k4[4] = {0.f, 0.f, 0.f, 0.f};
        if (per == 4) { float4 v = kv_load4(kt); k4[0] = v.x; k4[1] = v.y; k4[2] = v.z; k4[3] = v.w; }
        else { float2 v = kv_load2(kt); k4[0] = v.x; k4[1] = v.y; }
        float kscale = KV_SCALE(ks, t0 + t, n_kv, kvh);
        #pragma unroll
        for (int g = 0; g < ATT_MAX_G; g++) {
            if (g < G) {
                float d = lane_dot(q[g], k4);
                for (int o = 16; o > 0; o >>= 1) d = __fadd_rn(d, __shfl_xor_sync(0xffffffff, d, o));
                if (lane == 0) sc[t][g] = __fmul_rn(d, kscale);
            }
        }
    }
    }
    __syncthreads();
    // softmax per head; every head in one pass over the positions (each thread's share and the block-wide
    // reductions are the same per head as one head at a time)
    float mxg[ATT_MAX_G], seg[ATT_MAX_G];
    #pragma unroll
    for (int g = 0; g < ATT_MAX_G; g++) mxg[g] = NEG_INF, seg[g] = 0.f;
    for (int t = tid; t < len; t += blockDim.x)
        #pragma unroll
        for (int g = 0; g < ATT_MAX_G; g++) if (g < G) mxg[g] = fmaxf(mxg[g], sc[t][g]);
    #pragma unroll
    for (int g = 0; g < ATT_MAX_G; g++) if (g < G) mxg[g] = block_max(mxg[g], red);
    for (int t = tid; t < len; t += blockDim.x) {
        float vscale = KV_SCALE(vs, t0 + t, n_kv, kvh); // the value scale rides on the weight
        #pragma unroll
        for (int g = 0; g < ATT_MAX_G; g++) {
            if (g < G) {
                float e = expf(__fsub_rn(sc[t][g], mxg[g]));
                sc[t][g] = __fmul_rn(e, vscale);
                seg[g] = __fadd_rn(seg[g], e);
            }
        }
    }
    #pragma unroll
    for (int g = 0; g < ATT_MAX_G; g++) if (g < G) seg[g] = block_sum(seg[g], red); // barrier: the weights are visible to every thread
    for (int d = tid; d < hd; d += blockDim.x) { // one pass for hd <= 128 (128 threads), two for 256
        float a[ATT_MAX_G];
        #pragma unroll
        for (int g = 0; g < ATT_MAX_G; g++) a[g] = 0.f;
        const kv_t* vcol = vc + (size_t)t0 * kv_dim + kvh * hd + d;
        // positions in order (the sums' order is fixed); unrolled so the loads of several go out together
        #pragma unroll 8
        for (int t = 0; t < len; t++) {
            float v = kv_load1(vcol + (size_t)t * kv_dim);
            float4 w0 = *reinterpret_cast<const float4*>(&sc[t][0]);
            float w[ATT_MAX_G] = {w0.x, w0.y, w0.z, w0.w};
            if (G > 4) {
                float4 w1 = *reinterpret_cast<const float4*>(&sc[t][4]);
                w[4] = w1.x; w[5] = w1.y; w[6] = w1.z; w[7] = w1.w;
            }
            #pragma unroll
            for (int g = 0; g < ATT_MAX_G; g++) if (g < G) a[g] = __fmaf_rn(w[g], v, a[g]);
        }
        #pragma unroll
        for (int g = 0; g < ATT_MAX_G; g++) {
            if (g < G) {
                int h = kvh * G + g;
                if (nsplit == 1) {
                    out[(size_t)b * q_dim + h * hd + d] = __fmul_rn(a[g], __fdiv_rn(1.f, seg[g]));
                } else {
                    float* p = part + (((size_t)b * n_heads + h) * nsplit + s) * (hd + 2);
                    p[d] = a[g];
                    if (d == 0) { p[hd] = mxg[g]; p[hd + 1] = seg[g]; }
                }
            }
        }
    }
}

// Flash-decoding attention, part 1 (one head per block; used when attention_partial_gqa does not apply):
// one block per (head, token, slice of `chunk` positions). Writes the slice's unnormalised output plus
// its max score and exp-sum to `part` (hd + 2 floats).
// With nsplit == 1 there is nothing to merge: the block writes the normalised result straight to `out`.
__global__ void attention_partial(const float* __restrict__ qkv, const kv_t* __restrict__ kc, const kv_t* __restrict__ vc,
                                  const float* __restrict__ ks, const float* __restrict__ vs, float* __restrict__ part, float* __restrict__ out, int pos0, const int* __restrict__ posp, int n_heads, int n_kv, int hd,
                                  float scale, int nsplit, int want) {
    if (posp) pos0 = *posp;
    __shared__ float sc[ATT_MAX_CHUNK];
    __shared__ float qs[ATT_MAX_HD];
    __shared__ float red[32];
    __shared__ float accs[4][ATT_MAX_HD];
    int h = blockIdx.x, b = blockIdx.y, s = blockIdx.z;
    int span = pos0 + b + 1; // causal
    int chunk = att_chunk_for(span, want);
    int t0 = s * chunk, len = min(span, t0 + chunk) - t0;
    float* p = part + (((size_t)b * n_heads + h) * nsplit + s) * (hd + 2);
    if (len <= 0) { // same for the whole block, so returning before the barriers is safe
        for (int d = threadIdx.x; d < hd; d += blockDim.x) p[d] = 0.f;
        if (threadIdx.x == 0) { p[hd] = NEG_INF; p[hd + 1] = 0.f; }
        return;
    }
    int kvh = h / (n_heads / n_kv);
    int q_dim = n_heads * hd, kv_dim = n_kv * hd;
    const float* qh = qkv + (size_t)b * (q_dim + 2 * kv_dim) + h * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) qs[d] = qh[d];
    __syncthreads();

    float mx = NEG_INF;
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        const kv_t* kt = kc + (size_t)(t0 + t) * kv_dim + kvh * hd;
        float acc = 0.f;
        for (int d = 0; d < hd / 4; d++) {
            float4 k4 = kv_load4(kt + 4 * d);
            acc += qs[4 * d] * k4.x + qs[4 * d + 1] * k4.y + qs[4 * d + 2] * k4.z + qs[4 * d + 3] * k4.w;
        }
        acc *= scale * KV_SCALE(ks, t0 + t, n_kv, kvh);
        sc[t] = acc;
        mx = fmaxf(mx, acc);
    }
    mx = block_max(mx, red);
    float se = 0.f;
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        float e = expf(sc[t] - mx);
        sc[t] = e * KV_SCALE(vs, t0 + t, n_kv, kvh); // the value scale rides on the weight
        se += e;
    }
    se = block_sum(se, red); // barrier: all scores visible

    // threads form `groups` copies of the head dimension, each summing every groups-th position
    int groups = blockDim.x / hd;
    int g = threadIdx.x / hd, d = threadIdx.x % hd;
    if (g < groups) {
        float acc = 0.f;
        for (int t = g; t < len; t += groups) acc += sc[t] * kv_load1(vc + (size_t)(t0 + t) * kv_dim + kvh * hd + d);
        accs[g][d] = acc;
    }
    __syncthreads();
    if (threadIdx.x < hd) {
        float acc = 0.f;
        for (int k = 0; k < groups; k++) acc += accs[k][threadIdx.x];
        if (nsplit == 1) out[(size_t)b * q_dim + h * hd + threadIdx.x] = acc * (1.f / se); // as attention_combine
        else p[threadIdx.x] = acc;
    }
    if (threadIdx.x == 0 && nsplit > 1) { p[hd] = mx; p[hd + 1] = se; }
}

// FlashAttention-style kernel for prompts: one block per (head, tile of PQT query tokens). Key/value tiles
// of PKT positions are staged in shared memory once for all PQT queries, with a running (online) softmax
// per query. 128 threads: each owns 2 query rows (16 row pairs) x one of 8 lanes ("g"); for scores a thread
// covers keys g, g+8, g+16, g+24 of the tile, for outputs the dims g, g+8, ... (interleaved: no bank conflicts).
#define PQT 32
#define PKT 32
__global__ void attention_prefill(const float* __restrict__ qkv, const kv_t* __restrict__ kc, const kv_t* __restrict__ vc,
                                  const float* __restrict__ ks, const float* __restrict__ vs,
                                  float* __restrict__ out, int pos0, int n, int n_heads, int n_kv, int hd, float scale) {
    extern __shared__ float smem[];
    int ld = hd + 1; // padded row stride: rows land on different banks
    float* Ks = smem;
    float* Vs = Ks + PKT * ld;
    float* Qs = Vs + PKT * ld;
    float* S = Qs + PQT * ld; // PQT x (PKT + 1)
    const int sld = PKT + 1;
    int h = blockIdx.x, q0 = blockIdx.y * PQT;
    int kvh = h / (n_heads / n_kv);
    int q_dim = n_heads * hd, kv_dim = n_kv * hd, row_stride = q_dim + 2 * kv_dim;
    int tid = threadIdx.x, g = tid & 7;
    int r0 = (tid >> 3) * 2, r1 = r0 + 1;
    int dpt = hd >> 3; // output dims per thread (hd <= 128)
    int kv_len = pos0 + n;

    for (int idx = tid; idx < PQT * hd; idx += blockDim.x) {
        int r = idx / hd, d = idx % hd;
        Qs[r * ld + d] = (q0 + r < n) ? qkv[(size_t)(q0 + r) * row_stride + h * hd + d] : 0.f;
    }
    float m[2] = {NEG_INF, NEG_INF}, l[2] = {0.f, 0.f}, acc[2][16];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[0][i] = 0.f; acc[1][i] = 0.f; }
    int lim0 = pos0 + q0 + r0, lim1 = lim0 + 1; // causal limits of the two rows
    int kmax = min(kv_len, pos0 + q0 + PQT);
    for (int t0 = 0; t0 < kmax; t0 += PKT) {
        __syncthreads(); // previous tile consumed (and Qs complete, the first time)
        for (int idx = tid; idx < PKT * hd; idx += blockDim.x) {
            int k = idx / hd, d = idx % hd, kp = t0 + k;
            bool ok = kp < kv_len;
            Ks[k * ld + d] = ok ? kv_load1(kc + (size_t)kp * kv_dim + kvh * hd + d) * KV_SCALE(ks, kp, n_kv, kvh) : 0.f;
            Vs[k * ld + d] = ok ? kv_load1(vc + (size_t)kp * kv_dim + kvh * hd + d) * KV_SCALE(vs, kp, n_kv, kvh) : 0.f;
        }
        __syncthreads();

        // scores for 2 rows x 4 keys, operands from registers: 6 shared reads per 8 FMAs
        float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        for (int d = 0; d < hd; d++) {
            float qa = Qs[r0 * ld + d], qb = Qs[r1 * ld + d];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float kv = Ks[(g + 8 * j) * ld + d];
                s[0][j] += qa * kv;
                s[1][j] += qb * kv;
            }
        }
        // online softmax per row; the row's 8 threads are adjacent lanes of one warp
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int lim = i ? lim1 : lim0, r = i ? r1 : r0;
            float mx = NEG_INF;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int kp = t0 + g + 8 * j;
                s[i][j] = (kp < kv_len && kp <= lim) ? s[i][j] * scale : NEG_INF;
                mx = fmaxf(mx, s[i][j]);
            }
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
            float m_new = fmaxf(m[i], mx);
            float alpha = (m[i] == NEG_INF) ? 0.f : expf(m[i] - m_new);
            float ls = 0.f;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float pv = (s[i][j] == NEG_INF) ? 0.f : expf(s[i][j] - m_new);
                S[r * sld + g + 8 * j] = pv;
                ls += pv;
            }
            for (int o = 1; o < 8; o <<= 1) ls += __shfl_xor_sync(0xffffffff, ls, o);
            l[i] = l[i] * alpha + ls;
            m[i] = m_new;
            #pragma unroll
            for (int dd = 0; dd < 16; dd++) acc[i][dd] *= alpha;
        }
        __syncwarp(); // the rows' probabilities, written by their 8 lanes, are read by all 8

        // out += P V: per key 2 probabilities + dpt values for 2*dpt FMAs
        for (int k = 0; k < PKT; k++) {
            float pa = S[r0 * sld + k], pb = S[r1 * sld + k];
            const float* vr = Vs + k * ld + g;
            #pragma unroll
            for (int dd = 0; dd < 16; dd++) {
                if (dd < dpt) {
                    float v = vr[8 * dd];
                    acc[0][dd] += pa * v;
                    acc[1][dd] += pb * v;
                }
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        int r = i ? r1 : r0;
        if (q0 + r < n) {
            float inv = 1.f / l[i];
            for (int dd = 0; dd < dpt; dd++) out[(size_t)(q0 + r) * q_dim + h * hd + g + 8 * dd] = acc[i][dd] * inv;
        }
    }
}

}  // extern "C": the template below needs C++ linkage

#ifdef KV8
// Decode attention for a verify batch: attention_partial_gqa's results for each token, bit for bit, at the cost
// of tiled matrix products. Grouped-query models, int8 cache, head size 128. Two kernels, each light enough on
// shared memory and registers that several blocks share an SM: attention_vfy_s (scores, then the softmax) and
// attention_vfy_o (outputs). They pass scores, then weights, through `scr`: a row of ATT_MAX_CHUNK floats per
// (token, head, slice), from the token's slice start. A block takes one (kv head, slice s, group of NB tokens,
// NB G <= 16 (token, head) pairs); each token keeps its own slice (att_chunk_for of its own span), and the
// group's tokens share every staged key and value row (rows padded by one word: lanes reading different rows
// hit different banks). 256 threads, 8 warps; for scores warp w owns pairs 4 (w % 4) .. + 3 and half w / 4 of
// a tile's positions.
// Scores: lane holds positions lane + 32c (c = 2 half, 2 half + 1). A score is attention_partial_gqa's: 32 lane
// shares (lane_dot) summed as its xor butterfly sums them, a perfect tree over the shares in bit-reversed order,
// built with a binary-counter stack in registers. The exp-sum is that kernel's 128 thread partials and
// block_sum's trees (warp_sum on the same layout); the max is order-free. Outputs (128 threads): warp w owns
// pairs 4w .. 4w + 3 and lane dimensions 4 lane .. + 3, from value rows converted to floats once per tile and the
// 4 pairs' weights side by side (one broadcast read); each is the same chain over the slice in order (positions
// outside a token's slice are skipped, not weighted by zero).
#define AV_TILE 128 // key rows per staged tile
#define AV_VO 32    // value rows per staged tile (as floats)
#define AV_WS 20    // floats per staged position of weights: 16 pairs (+4: the stores' banks spread)
#define AV_RW 33    // words per staged row of 128 int8 values

// 4 int8 values of a word to floats, exactly, at full rate: each byte xor 0x80 (so 0..255) goes into the low
// mantissa bits of 2^23, then 2^23 + 128 comes off
__device__ __forceinline__ void i8x4_to_f(int w, float* f) {
    unsigned u = (unsigned)w ^ 0x80808080u;
    #pragma unroll
    for (int k = 0; k < 4; k++) f[k] = __fsub_rn(__int_as_float(__byte_perm(u, 0x4B000000u, 0x7650 + k)), 8388736.f);
}

// A group's slices, as every thread of both kernels sees them.
template <int NB>
struct AvSlices {
    int t0[NB], len[NB], u0, u1; // each token's slice; the positions any of them covers
    __device__ __forceinline__ AvSlices(int pos0, int b0, int n, int s, int want) {
        u0 = 0x7fffffff;
        u1 = 0;
        #pragma unroll
        for (int b = 0; b < NB; b++) {
            t0[b] = 0;
            len[b] = 0;
            if (b0 + b < n) {
                int span = pos0 + b0 + b + 1; // causal
                int chunk = att_chunk_for(span, want);
                t0[b] = s * chunk;
                int l = min(span, t0[b] + chunk) - t0[b];
                if (l > 0) {
                    len[b] = l;
                    u0 = min(u0, t0[b]);
                    u1 = max(u1, t0[b] + l);
                }
            }
        }
    }
    // pair pr's token and slice [lo, hi) (absolute positions; empty for pr >= NB G); unrolled selects: no local memory
    __device__ __forceinline__ void pair(int pr, int G, int* b, int* lo, int* hi) const {
        *b = 0;
        *lo = 0;
        *hi = 0;
        #pragma unroll
        for (int bb = 0; bb < NB; bb++)
            if (pr >= bb * G && pr < (bb + 1) * G) { *b = bb; *lo = t0[bb]; *hi = t0[bb] + len[bb]; }
    }
};

// the (token, head, slice) row of a pair, in part (hd + 2 floats) or scr (ATT_MAX_CHUNK floats)
__device__ __forceinline__ size_t av_row(int tok, int n_heads, int h, int nsplit, int s) {
    return ((size_t)tok * n_heads + h) * nsplit + s;
}

template <int NB>
__device__ __forceinline__ void att_vfy_s(const float* __restrict__ qkv, const kv_t* __restrict__ kc, const float* __restrict__ ks,
                                          const float* __restrict__ vs, float* __restrict__ part, float* __restrict__ scr, int pos0,
                                          int n_heads, int n_kv, float scale, int nsplit, int want, int n) {
    const int hd = 128;
    __shared__ __align__(16) float sq[16 * 128];     // [pair][hd]: queries, times scale
    __shared__ __align__(16) int skv[AV_TILE * AV_RW]; // a tile of key rows
    __shared__ float sks[AV_TILE];                   // their scales
    int kvh = blockIdx.x, s = blockIdx.y, b0 = blockIdx.z * NB;
    int G = n_heads / n_kv, P = NB * G; // pair i is token i / G, head i % G
    int q_dim = n_heads * hd, kv_dim = n_kv * hd;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, half = warp >> 2;
    AvSlices<NB> sl(pos0, b0, n, s, want);
    #pragma unroll
    for (int b = 0; b < NB; b++) {
        if (b0 + b < n && sl.len[b] == 0) {
            // nothing in this slice: an empty part, as attention_combine expects
            for (int g = 0; g < G; g++) {
                float* pp = part + av_row(b0 + b, n_heads, kvh * G + g, nsplit, s) * (hd + 2);
                for (int d = tid; d < hd; d += blockDim.x) pp[d] = 0.f;
                if (tid == 0) { pp[hd] = NEG_INF; pp[hd + 1] = 0.f; }
            }
        }
    }
    if (sl.u1 <= sl.u0) return; // the same for the whole block: safe before the barriers
    for (int pr = 0; pr < P; pr++) {
        int b, lo, hi;
        sl.pair(pr, G, &b, &lo, &hi);
        const float* q = qkv + (size_t)min(b0 + b, n - 1) * (q_dim + 2 * kv_dim) + (kvh * G + pr - b * G) * hd;
        if (tid < hd) sq[pr * hd + tid] = __fmul_rn(q[tid], scale);
    }
    int w4 = 4 * (warp & 3); // this warp's pairs
    int plo[4], phi[4];
    float* prow[4]; // a pair's scores by absolute position
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int b;
        sl.pair(w4 + j, G, &b, &plo[j], &phi[j]);
        prow[j] = scr + av_row(b0 + b, n_heads, kvh * G + w4 + j - b * G, nsplit, s) * ATT_MAX_CHUNK - plo[j];
    }
    // a key tile is 4 pieces of 16 bytes per thread (and a scale for the first AV_TILE threads), fetched into
    // registers a tile ahead so the loads are in flight while the last tile is scored
    int4 kn[4];
    float ksn = 0.f;
    auto fetch = [&](int p0) {
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int i = tid + 256 * k, rr = i >> 3;
            if (p0 + rr < sl.u1) kn[k] = *reinterpret_cast<const int4*>(kc + (size_t)(p0 + rr) * kv_dim + kvh * hd + 16 * (i & 7));
        }
        if (tid < AV_TILE && p0 + tid < sl.u1) ksn = KV_SCALE(ks, p0 + tid, n_kv, kvh);
    };
    fetch(sl.u0);
    for (int p0 = sl.u0; p0 < sl.u1; p0 += AV_TILE) {
        int tl = min(AV_TILE, sl.u1 - p0);
        __syncthreads(); // the queries are in; the last tile is done with
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int i = tid + 256 * k, rr = i >> 3;
            if (rr < tl) {
                int* dst = skv + rr * AV_RW + 4 * (i & 7);
                dst[0] = kn[k].x; dst[1] = kn[k].y; dst[2] = kn[k].z; dst[3] = kn[k].w;
            }
        }
        if (tid < tl) sks[tid] = ksn;
        __syncthreads();
        if (p0 + AV_TILE < sl.u1) fetch(p0 + AV_TILE);
        if (w4 < P && 64 * half < tl) { // a half of the tile with no rows left: nothing to score
            float st[4][2][5], res[4][2];
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                const int l = ((i & 1) << 4) | ((i & 2) << 2) | (i & 4) | ((i & 8) >> 2) | ((i & 16) >> 4); // lane bit-reversed
                float qv[4][4], kf[2][4];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    float4 a = *reinterpret_cast<const float4*>(sq + (w4 + j) * hd + 4 * l);
                    qv[j][0] = a.x; qv[j][1] = a.y; qv[j][2] = a.z; qv[j][3] = a.w;
                }
                #pragma unroll
                for (int c = 0; c < 2; c++) i8x4_to_f(skv[(lane + 32 * (2 * half + c)) * AV_RW + l], kf[c]);
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    #pragma unroll
                    for (int c = 0; c < 2; c++) {
                        float x = lane_dot(qv[j], kf[c]);
                        // binary counter: slot k holds the 2^k shares before this one when bit k of i is set
                        #pragma unroll
                        for (int k = 0; k < 5; k++) {
                            if (i & (1 << k)) {
                                x = __fadd_rn(st[j][c][k], x);
                            } else {
                                st[j][c][k] = x;
                                break;
                            }
                        }
                        if (i == 31) res[j][c] = x;
                    }
            }
            #pragma unroll
            for (int j = 0; j < 4; j++)
                #pragma unroll
                for (int c = 0; c < 2; c++) {
                    int r = lane + 32 * (2 * half + c), tp = p0 + r;
                    if (r < tl && tp >= plo[j] && tp < phi[j]) prow[j][tp] = __fmul_rn(res[j][c], sks[r]);
                }
        }
    }
    __syncthreads(); // the block's scores are in
    // each pair's max, weights and exp-sum, as attention_partial_gqa's 128 threads and block_sum form them
    for (int pr = warp; pr < P; pr += 8) {
        int b, lo, hi;
        sl.pair(pr, G, &b, &lo, &hi);
        int L = hi - lo, h = kvh * G + pr - b * G;
        if (L <= 0) continue;
        float* w = scr + av_row(b0 + b, n_heads, h, nsplit, s) * ATT_MAX_CHUNK;
        float m = NEG_INF;
        for (int x = lane; x < L; x += 32) m = fmaxf(m, w[x]);
        m = __shfl_sync(0xffffffff, warp_max(m), 0);
        float part4[4];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float se = 0.f;
            for (int x = 32 * k + lane; x < L; x += 128) {
                float e = expf(__fsub_rn(w[x], m));
                w[x] = __fmul_rn(e, KV_SCALE(vs, lo + x, n_kv, kvh));
                se = __fadd_rn(se, e);
            }
            part4[k] = __shfl_sync(0xffffffff, warp_sum(se), 0); // thread block k's warp sum
        }
        float tot = warp_sum(lane < 4 ? (lane == 0 ? part4[0] : lane == 1 ? part4[1] : lane == 2 ? part4[2] : part4[3]) : 0.f);
        if (lane == 0) {
            float* pp = part + av_row(b0 + b, n_heads, h, nsplit, s) * (hd + 2);
            pp[hd] = m;
            pp[hd + 1] = tot;
        }
    }
}

template <int NB>
__device__ __forceinline__ void att_vfy_o(const kv_t* __restrict__ vc, float* __restrict__ part, float* __restrict__ out,
                                          const float* __restrict__ scr, int pos0, int n_heads, int n_kv, int nsplit, int want, int n) {
    const int hd = 128, ld = hd + 4; // value rows as floats, padded to 16-byte rows
    __shared__ __align__(16) float sv[AV_VO * (128 + 4)]; // a tile of value rows, converted once
    __shared__ __align__(16) float sw[AV_VO * AV_WS];     // [tile position][pair]: weights, a warp's 4 side by side
    int kvh = blockIdx.x, s = blockIdx.y, b0 = blockIdx.z * NB;
    int G = n_heads / n_kv, P = NB * G;
    int q_dim = n_heads * hd, kv_dim = n_kv * hd;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    AvSlices<NB> sl(pos0, b0, n, s, want);
    if (sl.u1 <= sl.u0) return;
    // warp w: pairs 4w .. 4w + 3; lane: dimensions 4 lane .. + 3
    int p4 = 4 * warp, bj[4], lo[4], hi[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) sl.pair(p4 + j, G, &bj[j], &lo[j], &hi[j]);
    bool one = lo[0] == lo[1] && lo[0] == lo[2] && lo[0] == lo[3] && hi[0] == hi[1] && hi[0] == hi[2] && hi[0] == hi[3];
    // a tile: two 16-byte pieces of value rows and 4 of the P x AV_VO weights per thread, fetched into registers a
    // tile ahead so the loads are in flight while the last tile is used
    int4 vn[2];
    float wn[4];
    auto fetch = [&](int p0) {
        #pragma unroll
        for (int k = 0; k < 2; k++) {
            int i = tid + 128 * k, rr = i >> 3;
            if (p0 + rr < sl.u1) vn[k] = *reinterpret_cast<const int4*>(vc + (size_t)(p0 + rr) * kv_dim + kvh * hd + 16 * (i & 7));
        }
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int i = tid + 128 * k, pr = i / AV_VO, tp = p0 + i % AV_VO;
            wn[k] = 0.f;
            if (pr < P) {
                int b, l0, h0;
                sl.pair(pr, G, &b, &l0, &h0);
                if (tp >= l0 && tp < h0) wn[k] = scr[av_row(b0 + b, n_heads, kvh * G + pr - b * G, nsplit, s) * ATT_MAX_CHUNK + tp - l0];
            }
        }
    };
    fetch(sl.u0);
    float acc[4][4];
    #pragma unroll
    for (int j = 0; j < 4; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) acc[j][e] = 0.f;
    for (int p0 = sl.u0; p0 < sl.u1; p0 += AV_VO) {
        int tl = min(AV_VO, sl.u1 - p0);
        __syncthreads(); // the last tile is done with
        #pragma unroll
        for (int k = 0; k < 2; k++) {
            int i = tid + 128 * k, rr = i >> 3;
            if (rr < tl) {
                float f[16];
                i8x4_to_f(vn[k].x, f);
                i8x4_to_f(vn[k].y, f + 4);
                i8x4_to_f(vn[k].z, f + 8);
                i8x4_to_f(vn[k].w, f + 12);
                float4* dst = reinterpret_cast<float4*>(sv + rr * ld + 16 * (i & 7));
                #pragma unroll
                for (int q = 0; q < 4; q++) dst[q] = make_float4(f[4 * q], f[4 * q + 1], f[4 * q + 2], f[4 * q + 3]);
            }
        }
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int i = tid + 128 * k;
            sw[(i % AV_VO) * AV_WS + i / AV_VO] = wn[k];
        }
        __syncthreads();
        if (p0 + AV_VO < sl.u1) fetch(p0 + AV_VO);
        if (p4 < P) {
            if (one) {
                int from = max(p0, lo[0]), to = min(p0 + tl, hi[0]);
                #pragma unroll 2
                for (int tp = from; tp < to; tp++) {
                    float4 v = *reinterpret_cast<const float4*>(sv + (tp - p0) * ld + 4 * lane);
                    float4 w = *reinterpret_cast<const float4*>(sw + (tp - p0) * AV_WS + p4);
                    float wv[4] = {w.x, w.y, w.z, w.w}, vv[4] = {v.x, v.y, v.z, v.w};
                    #pragma unroll
                    for (int j = 0; j < 4; j++)
                        #pragma unroll
                        for (int e = 0; e < 4; e++) acc[j][e] = __fmaf_rn(wv[j], vv[e], acc[j][e]);
                }
            } else {
                for (int tp = p0; tp < p0 + tl; tp++) {
                    float4 v = *reinterpret_cast<const float4*>(sv + (tp - p0) * ld + 4 * lane);
                    float4 w = *reinterpret_cast<const float4*>(sw + (tp - p0) * AV_WS + p4);
                    float wv[4] = {w.x, w.y, w.z, w.w}, vv[4] = {v.x, v.y, v.z, v.w};
                    #pragma unroll
                    for (int j = 0; j < 4; j++)
                        if (tp >= lo[j] && tp < hi[j])
                            #pragma unroll
                            for (int e = 0; e < 4; e++) acc[j][e] = __fmaf_rn(wv[j], vv[e], acc[j][e]);
                }
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int pr = p4 + j;
        if (pr >= P || hi[j] <= lo[j]) continue;
        int h = kvh * G + pr - bj[j] * G, tok = b0 + bj[j];
        float* pp = part + av_row(tok, n_heads, h, nsplit, s) * (hd + 2); // its max and exp-sum: attention_vfy_s's
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            int d = 4 * lane + e;
            if (nsplit == 1) out[(size_t)tok * q_dim + h * hd + d] = __fmul_rn(acc[j][e], __fdiv_rn(1.f, pp[hd + 1]));
            else pp[d] = acc[j][e];
        }
    }
}
#define ATT_VFY_S_PARAMS const float* __restrict__ qkv, const kv_t* __restrict__ kc, const float* __restrict__ ks, \
                         const float* __restrict__ vs, float* __restrict__ part, float* __restrict__ scr, int pos0, int n_heads, \
                         int n_kv, float scale, int nsplit, int want, int n
#define ATT_VFY_S_ARGS qkv, kc, ks, vs, part, scr, pos0, n_heads, n_kv, scale, nsplit, want, n
#define ATT_VFY_O_PARAMS const kv_t* __restrict__ vc, float* __restrict__ part, float* __restrict__ out, const float* __restrict__ scr, \
                         int pos0, int n_heads, int n_kv, int nsplit, int want, int n
#define ATT_VFY_O_ARGS vc, part, out, scr, pos0, n_heads, n_kv, nsplit, want, n
extern "C" __global__ void __launch_bounds__(256, 2) attention_vfy_s2(ATT_VFY_S_PARAMS) { att_vfy_s<2>(ATT_VFY_S_ARGS); }
extern "C" __global__ void __launch_bounds__(256, 2) attention_vfy_s4(ATT_VFY_S_PARAMS) { att_vfy_s<4>(ATT_VFY_S_ARGS); }
extern "C" __global__ void __launch_bounds__(256, 2) attention_vfy_s8(ATT_VFY_S_PARAMS) { att_vfy_s<8>(ATT_VFY_S_ARGS); }
extern "C" __global__ void __launch_bounds__(128, 3) attention_vfy_o2(ATT_VFY_O_PARAMS) { att_vfy_o<2>(ATT_VFY_O_ARGS); }
extern "C" __global__ void __launch_bounds__(128, 3) attention_vfy_o4(ATT_VFY_O_PARAMS) { att_vfy_o<4>(ATT_VFY_O_ARGS); }
extern "C" __global__ void __launch_bounds__(128, 3) attention_vfy_o8(ATT_VFY_O_PARAMS) { att_vfy_o<8>(ATT_VFY_O_ARGS); }
#endif

// two floats -> one f16x2 word (low half first); sm_75 has no packed f32->f16x2 convert
__device__ __forceinline__ unsigned pack_h2(float lo, float hi) {
    unsigned r;
    asm("{.reg .f16 l, h; cvt.rn.f16.f32 l, %1; cvt.rn.f16.f32 h, %2; mov.b32 %0, {l, h};}" : "=r"(r) : "f"(lo), "f"(hi));
    return r;
}
#ifdef KV8
// 4 int8 values (one word) to 4 exact f16 (two f16x2 words): with the sign bit flipped each byte u is
// 0..255, the f16 with bits 0x6400 | u is 1024 + u, and subtracting 1152 (0x6480) leaves the int8 value.
__device__ __forceinline__ uint2 i8x4_to_h4(unsigned w) {
    unsigned x = w ^ 0x80808080u, lo, hi;
    asm("prmt.b32 %0, %1, %2, 0x4140;" : "=r"(lo) : "r"(x), "r"(0x64646464u));
    asm("prmt.b32 %0, %1, %2, 0x4342;" : "=r"(hi) : "r"(x), "r"(0x64646464u));
    asm("sub.f16x2 %0, %0, %1;" : "+r"(lo) : "r"(0x64806480u));
    asm("sub.f16x2 %0, %0, %1;" : "+r"(hi) : "r"(0x64806480u));
    return make_uint2(lo, hi);
}
#endif
#define MMA_F16(c, a0, a1, b) \
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};" \
        : "+f"((c)[0]), "+f"((c)[1]), "+f"((c)[2]), "+f"((c)[3]) : "r"(a0), "r"(a1), "r"(b))
#define LDSM_X4(r, addr, trans) \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4" trans ".shared.b16 {%0,%1,%2,%3}, [%4];" \
                 : "=r"((r)[0]), "=r"((r)[1]), "=r"((r)[2]), "=r"((r)[3]) : "r"(addr))

// Tensor-core version of attention_prefill (sm_75+, FlashAttention-2 style): one block per (head, 64
// query tokens), 4 warps of 16 query rows. Q, K and V go through f16 m16n8k8 mma with f32 accumulators:
// a warp keeps its Q rows as fragments in registers, S = Q K^T comes out in the accumulator layout,
// which is also the A-fragment layout, so the probabilities feed O += P V without a trip through
// shared memory. K/V tiles of 32 positions are copied (f16, as cached) into shared memory (rows padded by 16
// bytes: ldmatrix rows hit different banks); the same buffer stages Q at the start. An int8 cache stages
// K and V as their exact integer values (in f16) and applies the scales to the scores and probabilities. Online softmax in
// base 2 (Q is pre-scaled by scale * log2 e). The row sums come from the tensor cores too (P x ones), so they\n// add up exactly the rounded probabilities that P V uses.
template <int HD, int SPLIT = 1> // SPLIT: blocks per head (grid z), each with HD / SPLIT of the output dimensions
__device__ void attn_prefill_tc(const float* __restrict__ qkv, const kv_t* __restrict__ kc, const kv_t* __restrict__ vc,
                                const float* __restrict__ ks, const float* __restrict__ vs,
                                float* __restrict__ out, int pos0, int n, int n_heads, int n_kv, float scale) {
    constexpr int BQ = 64, BK = 32, LDW = HD / 2 + 4; // row stride in f16x2 words
    __shared__ __align__(16) unsigned sm[2 * BK * LDW]; // Q staging (BQ rows), then K (BK rows) | V (BK rows)
#ifdef KV8
    // the tile's scales; the value scales go into the f16 probabilities times 2^10, so small products
    // stay normal f16 (the output is divided by it again); |V| up to ~8000 fits
    constexpr float VSCALE_BIAS = 1024.f;
    __shared__ float kscale[BK], vscale[BK];
#endif
    unsigned* Ks = sm;
    unsigned* Vs = sm + BK * LDW;
    int h = blockIdx.x;
    int q0 = (gridDim.y - 1 - blockIdx.y) * BQ; // the longest (latest) query tiles start first
    int kvh = h / (n_heads / n_kv);
    int q_dim = n_heads * HD, kv_dim = n_kv * HD, row_stride = q_dim + 2 * kv_dim;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int kv_len = pos0 + n;
    const float qs = scale * 1.4426950408889634f;

    // stage Q (rows past n are zeros), then each warp takes its 16 rows as A fragments
    for (int idx = tid; idx < BQ * HD / 4; idx += 128) {
        int r = idx / (HD / 4), c4 = idx % (HD / 4);
        float4 v = q0 + r < n ? *reinterpret_cast<const float4*>(qkv + (size_t)(q0 + r) * row_stride + h * HD + 4 * c4) : make_float4(0.f, 0.f, 0.f, 0.f);
        *reinterpret_cast<uint2*>(sm + r * LDW + 2 * c4) = make_uint2(pack_h2(v.x * qs, v.y * qs), pack_h2(v.z * qs, v.w * qs));
    }
    __syncthreads();
    unsigned qf[HD / 8][2]; // per k8 step: rows 0-7, rows 8-15
    {
        int m = lane >> 3; // ldmatrix.x4 matrices: (rows 0-7 | 8-15) x (k0 | k0+8)
        unsigned base = (unsigned)__cvta_generic_to_shared(sm + (16 * warp + (lane & 7) + 8 * (m & 1)) * LDW + 4 * (m >> 1));
        #pragma unroll
        for (int k = 0; k < HD / 16; k++) {
            unsigned r[4];
            LDSM_X4(r, base + 32 * k, "");
            qf[2 * k][0] = r[0]; qf[2 * k][1] = r[1]; qf[2 * k + 1][0] = r[2]; qf[2 * k + 1][1] = r[3];
        }
    }

    constexpr int DO = HD / SPLIT; // output dimensions of this block, from d0
    const int d0 = blockIdx.z * DO;
    float o[DO / 8][4]; // 16 rows x DO dims: per dim tile of 8, rows lo (c0, c1) and hi (c2, c3)
    #pragma unroll
    for (int d = 0; d < DO / 8; d++) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.f;
    float m_lo = NEG_INF, m_hi = NEG_INF;
    float l[4] = {0.f, 0.f, 0.f, 0.f}; // row sums of the f16 probabilities: P x ones (c0 = row lo, c2 = row hi)
    int p_lo = pos0 + q0 + 16 * warp + (lane >> 2), p_hi = p_lo + 8; // positions of this thread's two rows
    int warp_last = pos0 + q0 + 16 * warp + 15;
    int kmax = min(kv_len, pos0 + q0 + BQ);
    // ldmatrix row addresses: K (keys as rows, non-transposed = B fragments of K^T), V (transposed)
    unsigned kaddr = (unsigned)__cvta_generic_to_shared(Ks + (8 * 0 + (lane & 7)) * LDW + 4 * (lane >> 3));
    unsigned vaddr = (unsigned)__cvta_generic_to_shared(Vs + lane * LDW);

    for (int t0 = 0; t0 < kmax; t0 += BK) {
        __syncthreads(); // previous tile (or the Q staging) consumed
#ifdef KV8
        // int8 rows are HD bytes, so one 16-byte load brings 16 values: every load of the tile (and its
        // scales) is issued before any is used, then converted to exact f16 integers. The scales go on
        // the scores (K) and the probabilities (V). Past the end: zeros, so 0 x scale stays 0.
        constexpr int C16 = HD / 16, NL = BK * C16 / 128; // 16-value groups per row, loads per thread
        float sc_ld = 0.f;
        if (tid < 2 * BK && t0 + (tid & (BK - 1)) < kv_len)
            sc_ld = tid < BK ? KV_SCALE(ks, t0 + tid, n_kv, kvh) : KV_SCALE(vs, t0 + tid - BK, n_kv, kvh) * VSCALE_BIAS;
        uint4 kw[NL], vw[NL];
        #pragma unroll
        for (int i = 0; i < NL; i++) {
            int idx = tid + 128 * i, kp = t0 + idx / C16, off = 16 * (idx % C16);
            kw[i] = vw[i] = make_uint4(0u, 0u, 0u, 0u);
            if (kp < kv_len) {
                kw[i] = *reinterpret_cast<const uint4*>(kc + (size_t)kp * kv_dim + kvh * HD + off);
                vw[i] = *reinterpret_cast<const uint4*>(vc + (size_t)kp * kv_dim + kvh * HD + off);
            }
        }
        #pragma unroll
        for (int i = 0; i < NL; i++) {
            int idx = tid + 128 * i, r = idx / C16, w = 8 * (idx % C16); // f16x2 word offset in the row
            uint2 a = i8x4_to_h4(kw[i].x), b = i8x4_to_h4(kw[i].y), c = i8x4_to_h4(kw[i].z), d = i8x4_to_h4(kw[i].w);
            *reinterpret_cast<uint4*>(Ks + r * LDW + w) = make_uint4(a.x, a.y, b.x, b.y);
            *reinterpret_cast<uint4*>(Ks + r * LDW + w + 4) = make_uint4(c.x, c.y, d.x, d.y);
            a = i8x4_to_h4(vw[i].x); b = i8x4_to_h4(vw[i].y); c = i8x4_to_h4(vw[i].z); d = i8x4_to_h4(vw[i].w);
            *reinterpret_cast<uint4*>(Vs + r * LDW + w) = make_uint4(a.x, a.y, b.x, b.y);
            *reinterpret_cast<uint4*>(Vs + r * LDW + w + 4) = make_uint4(c.x, c.y, d.x, d.y);
        }
        if (tid < BK) kscale[tid] = sc_ld;
        else if (tid < 2 * BK) vscale[tid - BK] = sc_ld;
#else
        for (int idx = tid; idx < BK * HD / 4; idx += 128) {
            int r = idx / (HD / 4), c4 = idx % (HD / 4), kp = t0 + r;
            uint2 k4 = make_uint2(0u, 0u), v4 = k4; // positions past the end must be finite (zeros): P = 0 there
            if (kp < kv_len) { // the cache is f16 already: copied as is
                k4 = *reinterpret_cast<const uint2*>(kc + (size_t)kp * kv_dim + kvh * HD + 4 * c4);
                v4 = *reinterpret_cast<const uint2*>(vc + (size_t)kp * kv_dim + kvh * HD + 4 * c4);
            }
            *reinterpret_cast<uint2*>(Ks + r * LDW + 2 * c4) = k4;
            *reinterpret_cast<uint2*>(Vs + r * LDW + 2 * c4) = v4;
        }
#endif
        __syncthreads();
        if (t0 > warp_last) continue; // causal: nothing of this tile is visible to the warp's rows

        // S = Q K^T: 4 key tiles of 8; per key tile ldmatrix.x4 covers 4 k8 steps (32 dims)
        float s[BK / 8][4];
        #pragma unroll
        for (int j = 0; j < BK / 8; j++) {
            s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
            #pragma unroll
            for (int k = 0; k < HD / 32; k++) {
                unsigned b[4];
                LDSM_X4(b, kaddr + (8 * j * LDW + 16 * k) * 4, "");
                #pragma unroll
                for (int q = 0; q < 4; q++) MMA_F16(s[j], qf[4 * k + q][0], qf[4 * k + q][1], b[q]);
            }
        }
        // causal / length mask, online softmax (row max over the 4 lanes of a quad)
        float mx_lo = NEG_INF, mx_hi = NEG_INF;
        #pragma unroll
        for (int j = 0; j < BK / 8; j++)
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                int kp = t0 + 8 * j + 2 * (lane & 3) + e;
#ifdef KV8
                float kd = kscale[8 * j + 2 * (lane & 3) + e];
                s[j][e] *= kd;
                s[j][2 + e] *= kd;
#endif
                if (kp > p_lo || kp >= kv_len) s[j][e] = NEG_INF;
                if (kp > p_hi || kp >= kv_len) s[j][2 + e] = NEG_INF;
                mx_lo = fmaxf(mx_lo, s[j][e]);
                mx_hi = fmaxf(mx_hi, s[j][2 + e]);
            }
        #pragma unroll
        for (int x = 1; x < 4; x <<= 1) {
            mx_lo = fmaxf(mx_lo, __shfl_xor_sync(0xffffffff, mx_lo, x));
            mx_hi = fmaxf(mx_hi, __shfl_xor_sync(0xffffffff, mx_hi, x));
        }
        // every row sees position 0 in the first tile, so the running max is finite from then on
        float mn_lo = fmaxf(m_lo, mx_lo), mn_hi = fmaxf(m_hi, mx_hi);
        float a_lo = exp2f(m_lo - mn_lo), a_hi = exp2f(m_hi - mn_hi);
        m_lo = mn_lo; m_hi = mn_hi;
        unsigned pf[BK / 8][2]; // probabilities as A fragments, one per k8 step of keys
#ifdef KV8
        unsigned pv[BK / 8][2]; // the same times the value scales, for P V
#else
        unsigned (&pv)[BK / 8][2] = pf;
#endif
        #pragma unroll
        for (int j = 0; j < BK / 8; j++) {
            float p0 = exp2f(s[j][0] - mn_lo), p1 = exp2f(s[j][1] - mn_lo);
            float p2 = exp2f(s[j][2] - mn_hi), p3 = exp2f(s[j][3] - mn_hi);
            pf[j][0] = pack_h2(p0, p1);
            pf[j][1] = pack_h2(p2, p3);
#ifdef KV8
            float v0 = vscale[8 * j + 2 * (lane & 3)], v1 = vscale[8 * j + 2 * (lane & 3) + 1];
            pv[j][0] = pack_h2(p0 * v0, p1 * v1);
            pv[j][1] = pack_h2(p2 * v0, p3 * v1);
#endif
        }
        // normalise by the sum of the rounded probabilities that P V actually uses
        l[0] *= a_lo; l[1] *= a_lo; l[2] *= a_hi; l[3] *= a_hi;
        #pragma unroll
        for (int j = 0; j < BK / 8; j++) MMA_F16(l, pf[j][0], pf[j][1], 0x3C003C00u); // f16 1.0 x2
        // O = O * alpha + P V: per dim tile one ldmatrix.x4.trans gives the B fragments of all 4 key steps
        #pragma unroll
        for (int d = 0; d < DO / 8; d++) {
            o[d][0] *= a_lo; o[d][1] *= a_lo; o[d][2] *= a_hi; o[d][3] *= a_hi;
            unsigned b[4];
            LDSM_X4(b, vaddr + 16 * (d0 / 8 + d), ".trans");
            #pragma unroll
            for (int j = 0; j < BK / 8; j++) MMA_F16(o[d], pv[j][0], pv[j][1], b[j]);
        }
    }
    int r_lo = q0 + 16 * warp + (lane >> 2), r_hi = r_lo + 8;
#ifdef KV8
    float i_lo = (1.f / VSCALE_BIAS) / l[0], i_hi = (1.f / VSCALE_BIAS) / l[2];
#else
    float i_lo = 1.f / l[0], i_hi = 1.f / l[2];
#endif
    #pragma unroll
    for (int d = 0; d < DO / 8; d++) {
        int col = h * HD + d0 + 8 * d + 2 * (lane & 3);
        if (r_lo < n) *reinterpret_cast<float2*>(out + (size_t)r_lo * q_dim + col) = make_float2(o[d][0] * i_lo, o[d][1] * i_lo);
        if (r_hi < n) *reinterpret_cast<float2*>(out + (size_t)r_hi * q_dim + col) = make_float2(o[d][2] * i_hi, o[d][3] * i_hi);
    }
}

extern "C" {

#define ATT_TC_ARGS const float* __restrict__ qkv, const kv_t* __restrict__ kc, const kv_t* __restrict__ vc, \
                    const float* __restrict__ ks, const float* __restrict__ vs, \
                    float* __restrict__ out, int pos0, int n, int n_heads, int n_kv, int hd, float scale
__global__ void __launch_bounds__(128) attention_prefill_tc64(ATT_TC_ARGS) { attn_prefill_tc<64>(qkv, kc, vc, ks, vs, out, pos0, n, n_heads, n_kv, scale); }
__global__ void __launch_bounds__(128) attention_prefill_tc128(ATT_TC_ARGS) { attn_prefill_tc<128>(qkv, kc, vc, ks, vs, out, pos0, n, n_heads, n_kv, scale); }
// head size 256: two blocks per head, each with half the output dimensions (all of Q, O for 256 do not fit in registers)
__global__ void __launch_bounds__(128) attention_prefill_tc256(ATT_TC_ARGS) { attn_prefill_tc<256, 2>(qkv, kc, vc, ks, vs, out, pos0, n, n_heads, n_kv, scale); }

#ifdef KV8
// Decode attention for 256-wide grouped heads on the tensor cores (int8 cache, sm_75+): attention_partial_gqa's
// grid, slices and partial results (attention_combine merges them), so a token is attended the same way
// alone or in a verify batch. The G <= 8 query heads of the kv head are the rows of the m16n8k8 tiles (rows
// 8..15 empty), 8 positions the columns: S = Q K^T over 32 k8 steps, O += P V over 32 tiles of 8
// dimensions. Fragments come straight from the cache rows (int8 -> exact f16), no shared memory staging.
// Warp w takes the slice's 8-position tiles w, w + 4, ...; online softmax in base 2 (Q pre-scaled by
// scale * log2 e); the probabilities times the value scales (and 2^10, as in attn_prefill_tc) are P.
// The 4 warps' results merge in shared memory, warp by warp in order.
__global__ void __launch_bounds__(128) attention_decode_tc256(const float* __restrict__ qkv, const kv_t* __restrict__ kc, const kv_t* __restrict__ vc,
                                                              const float* __restrict__ ks, const float* __restrict__ vs,
                                                              float* __restrict__ part, float* __restrict__ out, int pos0, const int* __restrict__ posp,
                                                              int n_heads, int n_kv, int hd, float scale, int nsplit, int want) {
    constexpr int HD = 256;
    constexpr float VSCALE_BIAS = 1024.f, LN2 = 0.6931471805599453f;
    __shared__ float wo[2][HD / 8][32]; // a warp's results, [pair][tile][lane]: no bank conflicts
    __shared__ float wm[8], wl[8];
    if (posp) pos0 = *posp;
    int kvh = blockIdx.x, b = blockIdx.y, s = blockIdx.z;
    int G = n_heads / n_kv;
    int span = pos0 + b + 1;
    int chunk = att_chunk_for(span, want);
    int t0 = s * chunk, len = min(span, t0 + chunk) - t0;
    int q_dim = n_heads * HD, kv_dim = n_kv * HD;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int g = lane >> 2, q = lane & 3; // this thread's row (head) and column pair
    if (len <= 0) {
        for (int gg = 0; gg < G; gg++) {
            float* p = part + (((size_t)b * n_heads + kvh * G + gg) * nsplit + s) * (HD + 2);
            for (int d = tid; d < HD; d += blockDim.x) p[d] = 0.f;
            if (tid == 0) { p[HD] = NEG_INF; p[HD + 1] = 0.f; }
        }
        return;
    }
    const float qs = scale * 1.4426950408889634f;
    // The dot products and P V do not depend on which tensor-core column handles which dimension, so the
    // dimensions are spread over the threads so that a warp reads whole cache sectors: in k8 step
    // k, thread q has a pair of the 16 bytes at 64i + 16q of the row (i = k / 8; Q and K alike); in output
    // tile d, column n is dimension 128(d / 16) + 16n + d % 16. Neighbouring lanes read neighbouring 16 bytes.
    unsigned qf[HD / 8]; // Q as A fragments (rows 8..15 are zeros, passed as 0)
    {
        const float* qrow = qkv + (size_t)b * (q_dim + 2 * kv_dim) + (kvh * G + min(g, G - 1)) * HD + 16 * q;
        #pragma unroll
        for (int k = 0; k < HD / 8; k++) { int dk = 64 * (k / 8) + 2 * (k % 8); qf[k] = g < G ? pack_h2(qrow[dk] * qs, qrow[dk + 1] * qs) : 0u; }
    }
    float o[HD / 8][2];
    #pragma unroll
    for (int d = 0; d < HD / 8; d++) o[d][0] = o[d][1] = 0.f;
    float m = NEG_INF, l = 0.f, z[2] = {0.f, 0.f}; // z: the empty rows' accumulators (stay 0)
    const kv_t* kbase = kc + kvh * HD;
    const kv_t* vbase = vc + kvh * HD;
    for (int tt = 8 * warp; tt < len; tt += 32) {
        // S: column g of the tile is position tt + g; this thread's 4 x 16 bytes of its key row
        float sc[2] = {0.f, 0.f};
        {
            const uint4* krow = reinterpret_cast<const uint4*>(kbase + (size_t)(t0 + min(tt + g, len - 1)) * kv_dim + 16 * q);
            uint4 kw[4] = {krow[0], krow[4], krow[8], krow[12]}; // bytes 64i + 16q
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                unsigned w = j % 4 == 0 ? kw[j / 4].x : j % 4 == 1 ? kw[j / 4].y : j % 4 == 2 ? kw[j / 4].z : kw[j / 4].w;
                uint2 e = i8x4_to_h4(w); // k8 steps 2j, 2j + 1
                float c4[4] = {sc[0], sc[1], z[0], z[1]};
                MMA_F16(c4, qf[2 * j], 0u, e.x);
                MMA_F16(c4, qf[2 * j + 1], 0u, e.y);
                sc[0] = c4[0]; sc[1] = c4[1];
            }
        }
        // this thread's scores: row g, positions tt + 2q, +1
        int pa = tt + 2 * q;
        float s0 = pa < len ? sc[0] * KV_SCALE(ks, t0 + pa, n_kv, kvh) : NEG_INF;
        float s1 = pa + 1 < len ? sc[1] * KV_SCALE(ks, t0 + pa + 1, n_kv, kvh) : NEG_INF;
        float mx = fmaxf(s0, s1);
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2)); // position tt is in every tile: finite
        float mn = fmaxf(m, mx), alpha = exp2f(m - mn);
        float p0 = exp2f(s0 - mn), p1 = exp2f(s1 - mn);
        m = mn;
        l = l * alpha + p0 + p1;
        float v0 = pa < len ? KV_SCALE(vs, t0 + pa, n_kv, kvh) * VSCALE_BIAS : 0.f;
        float v1 = pa + 1 < len ? KV_SCALE(vs, t0 + pa + 1, n_kv, kvh) * VSCALE_BIAS : 0.f;
        unsigned pf = pack_h2(p0 * v0, p1 * v1); // A fragment: row g, positions 2q, 2q + 1
        // V: B fragment of tile d = rows pa, pa + 1 at 128(d / 16) + 16g + d % 16: 2 x 16 bytes of each
        const uint4* va = reinterpret_cast<const uint4*>(vbase + (size_t)(t0 + min(pa, len - 1)) * kv_dim + 16 * g);
        const uint4* vb = reinterpret_cast<const uint4*>(vbase + (size_t)(t0 + min(pa + 1, len - 1)) * kv_dim + 16 * g);
        uint4 a4[2] = {va[0], va[8]}, b4[2] = {vb[0], vb[8]}; // bytes 128i + 16g
        #pragma unroll
        for (int d = 0; d < HD / 8; d++) { o[d][0] *= alpha; o[d][1] *= alpha; }
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            unsigned wa = j % 4 == 0 ? a4[j / 4].x : j % 4 == 1 ? a4[j / 4].y : j % 4 == 2 ? a4[j / 4].z : a4[j / 4].w;
            unsigned wb = j % 4 == 0 ? b4[j / 4].x : j % 4 == 1 ? b4[j / 4].y : j % 4 == 2 ? b4[j / 4].z : b4[j / 4].w;
            uint2 h01 = i8x4_to_h4(__byte_perm(wa, wb, 0x5140)); // (a0, b0), (a1, b1): tiles 4j, 4j + 1
            uint2 h23 = i8x4_to_h4(__byte_perm(wa, wb, 0x7362)); // (a2, b2), (a3, b3): tiles 4j + 2, 4j + 3
            unsigned bf[4] = {h01.x, h01.y, h23.x, h23.y};
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                float c4[4] = {o[4 * j + e][0], o[4 * j + e][1], z[0], z[1]};
                MMA_F16(c4, pf, 0u, bf[e]);
                o[4 * j + e][0] = c4[0]; o[4 * j + e][1] = c4[1];
            }
        }
    }
    l += __shfl_xor_sync(0xffffffff, l, 1);
    l += __shfl_xor_sync(0xffffffff, l, 2);
    // merge: warps 1..3 hand their state to warp 0 in turn
    for (int w = 1; w < 4; w++) {
        __syncthreads();
        if (warp == w) {
            if (q == 0) { wm[g] = m; wl[g] = l; }
            #pragma unroll
            for (int d = 0; d < HD / 8; d++) { wo[0][d][lane] = o[d][0]; wo[1][d][lane] = o[d][1]; }
        }
        __syncthreads();
        if (warp == 0 && wm[g] != NEG_INF) {
            float mw = wm[g], mn = fmaxf(m, mw), a = exp2f(m - mn), aw = exp2f(mw - mn);
            l = l * a + wl[g] * aw;
            #pragma unroll
            for (int d = 0; d < HD / 8; d++) {
                o[d][0] = o[d][0] * a + wo[0][d][lane] * aw;
                o[d][1] = o[d][1] * a + wo[1][d][lane] * aw;
            }
            m = mn;
        }
    }
    if (warp == 0 && g < G) {
        int h = kvh * G + g;
        bool whole = nsplit == 1;
        float f = whole ? (1.f / VSCALE_BIAS) / l : 1.f / VSCALE_BIAS;
        float* dst = whole ? out + (size_t)b * q_dim + h * HD : part + (((size_t)b * n_heads + h) * nsplit + s) * (HD + 2);
        #pragma unroll
        for (int d = 0; d < HD / 8; d++) { int dd = 128 * (d / 16) + 32 * q + d % 16; dst[dd] = o[d][0] * f; dst[dd + 16] = o[d][1] * f; }
        if (!whole && q == 0) { dst[HD] = m * LN2; dst[HD + 1] = l; } // natural-log units, as attention_combine expects
    }
}
#endif

// Flash-decoding attention, part 2: merge the slices of each (head, token) into the normalised output.
#define COMBINE_MAX 2048 // slices whose weights fit in shared memory (more: computed where used)
__global__ void attention_combine(const float* __restrict__ part, float* __restrict__ out,
                                  int pos0, const int* __restrict__ posp, int n_heads, int hd, int nsplit, int want) {
    // one warp per (head, token, 32 output dimensions): the slices' max score, each slice's weight
    // exp(max_s - max) once, the exp-sums weighted in slice order, then each dimension in slice order
    if (posp) pos0 = *posp;
    __shared__ float wsh[COMBINE_MAX];
    __shared__ float inv_sh;
    int h = blockIdx.x, b = blockIdx.y, lane = threadIdx.x;
    int span = pos0 + b + 1;
    int chunk = att_chunk_for(span, want);
    int used = min(nsplit, (span + chunk - 1) / chunk);
    const float* p = part + ((size_t)b * n_heads + h) * nsplit * (hd + 2);
    float m = NEG_INF;
    for (int s = lane; s < used; s += 32) m = fmaxf(m, p[s * (hd + 2) + hd]);
    m = __shfl_sync(0xffffffff, warp_max(m), 0); // warp_max leaves the whole max in lane 0 only
    bool held = used <= COMBINE_MAX;
    if (held)
        for (int s = lane; s < used; s += 32) wsh[s] = expf(__fsub_rn(p[s * (hd + 2) + hd], m));
    __syncwarp();
    auto weight = [&](int s) { return held ? wsh[s] : expf(__fsub_rn(p[s * (hd + 2) + hd], m)); };
    if (lane == 0) {
        float total = 0.f;
        for (int s = 0; s < used; s++) total = __fmaf_rn(p[s * (hd + 2) + hd + 1], weight(s), total);
        inv_sh = __fdiv_rn(1.f, total);
    }
    __syncwarp();
    int d = blockIdx.z * 32 + lane;
    if (d < hd) {
        float acc = 0.f;
        #pragma unroll 8
        for (int s = 0; s < used; s++) acc = __fmaf_rn(p[s * (hd + 2) + d], weight(s), acc);
        out[(size_t)b * n_heads * hd + h * hd + d] = __fmul_rn(acc, inv_sh);
    }
}

// Activations to int8 in blocks of 32 (one warp per block): q = round(x / d), d = max|x| / 127,
// plus each block's sum of q (for the K-quant "min" terms).
__global__ void quantize_q8(const float* __restrict__ x, signed char* __restrict__ q, float* __restrict__ d,
                            int* __restrict__ sums, int n_blocks) {
    int blk = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (blk >= n_blocks) return;
    float v = x[(size_t)blk * 32 + lane];
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    float dd = amax / 127.f;
    float inv = dd != 0.f ? 1.f / dd : 0.f;
    int qi = (int)roundf(v * inv);
    q[(size_t)blk * 32 + lane] = (signed char)qi;
    for (int o = 16; o > 0; o >>= 1) qi += __shfl_xor_sync(0xffffffff, qi, o);
    if (lane == 0) { d[blk] = dd; sums[blk] = qi; }
}

// Batched int8 matmul for prompts: out[t*out_stride + out_off + r] (+)= sum_k W[r][k] * X[t][k].
// Activations are int8 + scale per 32. Each weight 32-block becomes int8 values plus a scale for each
// 16-value half and a "min": Q8_0 (one scale, no min), Q4_K (values 0..15, scale and min),
// Q6_K (values -32..31, two scales). 64x64 output tile per 256-thread block; each thread owns
// 4 rows x 4 tokens; __dp4a does 4 int8 multiply-adds.
#define TILE 64
__global__ void matmul_tiled(int fmt, const unsigned char* __restrict__ wq, const float* __restrict__ wd,
                             const signed char* __restrict__ xq, const float* __restrict__ xd, const int* __restrict__ xsum,
                             float* __restrict__ out, int rows, int cols, int n, int accumulate, int out_stride, int out_off) {
    // one step covers up to 8 blocks of 32 (a whole K-quant super-block): fewer barriers and scale decodes
    __shared__ int ws[TILE][65]; // 64 ints = 256 int8 per row, +1 to spread rows over banks
    __shared__ int xs[TILE][65];
    __shared__ float wsa[TILE][8], wsb[TILE][8], wmn[TILE][8], xds[TILE][8], xss[TILE][8];
    int tid = threadIdx.x;
    int tx = tid & 15, ty = tid >> 4; // tx picks rows (coalesced stores), ty picks tokens
    int row0 = blockIdx.x * TILE, tok0 = blockIdx.y * TILE;
    int nb = cols >> 5, nsb = cols >> 8;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;

    for (int kc = 0; kc < nb; kc += 8) {
        int nk = min(8, nb - kc);
        for (int idx = tid; idx < TILE * 64; idx += 256) {
            int r = idx >> 6, wi = idx & 63, k8 = wi >> 3, w = wi & 7;
            int gr = row0 + r, gt = tok0 + r, kb = kc + k8;
            int v = 0, xv = 0;
            if (k8 < nk) {
                if (gr < rows) {
                    if (fmt == 0) {
                        v = *reinterpret_cast<const int*>(wq + (size_t)gr * cols + kb * 32 + w * 4);
                    } else if (fmt == 1) {
                        const unsigned char* b = wq + ((size_t)gr * nsb + (kb >> 3)) * Q4K_BYTES;
                        int p = *reinterpret_cast<const int*>(b + 16 + 32 * (k8 >> 1) + 4 * w);
                        v = (k8 & 1) ? ((p >> 4) & 0x0F0F0F0F) : (p & 0x0F0F0F0F);
                    } else {
                        const unsigned char* b = wq + ((size_t)gr * nsb + (kb >> 3)) * Q6K_BYTES;
                        int half = k8 >> 2, g = k8 & 3;
                        int p = *reinterpret_cast<const int*>(b + 64 * half + (g & 1) * 32 + 4 * w);
                        int lo = (g >= 2) ? ((p >> 4) & 0x0F0F0F0F) : (p & 0x0F0F0F0F);
                        int h = *reinterpret_cast<const int*>(b + 128 + 32 * half + 4 * w);
                        v = __vsub4(lo | (((h >> (2 * g)) & 0x03030303) << 4), 0x20202020); // bytewise -32
                    }
                }
                if (gt < n) xv = *reinterpret_cast<const int*>(xq + (size_t)gt * cols + kb * 32 + w * 4);
            }
            ws[r][wi] = v;
            xs[r][wi] = xv;
        }
        // per-32 scales: 64 rows x 8 blocks for the weights, the same for the activations
        for (int idx = tid; idx < TILE * 8; idx += 256) {
            int r = idx >> 3, k8 = idx & 7, kb = kc + k8;
            int gr = row0 + r, gt = tok0 + r;
            float sa = 0.f, sb = 0.f, mn = 0.f;
            if (k8 < nk && gr < rows) {
                if (fmt == 0) {
                    sa = sb = wd[(size_t)gr * nb + kb];
                } else if (fmt == 1) {
                    const unsigned char* b = wq + ((size_t)gr * nsb + (kb >> 3)) * Q4K_BYTES;
                    int sc, m;
                    q4k_scale_min(k8, b + 4, &sc, &m);
                    float d, dmin;
                    q4k_dm(b, &d, &dmin);
                    sa = sb = d * sc;
                    mn = dmin * m;
                } else {
                    const unsigned char* b = wq + ((size_t)gr * nsb + (kb >> 3)) * Q6K_BYTES;
                    float d = wd[(size_t)gr * nsb + (kb >> 3)];
                    const signed char* sc = (const signed char*)(b + 192 + 8 * (k8 >> 2) + 2 * (k8 & 3));
                    sa = d * sc[0];
                    sb = d * sc[1];
                }
            }
            wsa[r][k8] = sa; wsb[r][k8] = sb; wmn[r][k8] = mn;
            bool tok_ok = k8 < nk && gt < n;
            xds[r][k8] = tok_ok ? xd[(size_t)gt * nb + kb] : 0.f;
            xss[r][k8] = tok_ok ? (float)xsum[(size_t)gt * nb + kb] : 0.f;
        }
        __syncthreads();
        for (int k8 = 0; k8 < nk; k8++) {
            int wr[4][8], xr[4][8];
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int w = 0; w < 8; w++) { wr[i][w] = ws[tx + 16 * i][k8 * 8 + w]; xr[i][w] = xs[ty + 16 * i][k8 * 8 + w]; }
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int r = tx + 16 * i;
                float sa = wsa[r][k8], sb = wsb[r][k8], mn = wmn[r][k8];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int lo = 0, hi = 0;
                    #pragma unroll
                    for (int w = 0; w < 4; w++) lo = __dp4a(wr[i][w], xr[j][w], lo);
                    #pragma unroll
                    for (int w = 4; w < 8; w++) hi = __dp4a(wr[i][w], xr[j][w], hi);
                    int t = ty + 16 * j;
                    acc[i][j] += xds[t][k8] * (sa * (float)lo + sb * (float)hi - mn * xss[t][k8]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = row0 + tx + 16 * i;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int t = tok0 + ty + 16 * j;
            if (r < rows && t < n) {
                size_t idx = (size_t)t * out_stride + out_off + r;
                out[idx] = accumulate ? out[idx] + acc[i][j] : acc[i][j];
            }
        }
    }
}

// Tensor-core version of matmul_tiled (sm_75+): int8 mma.sync.m8n8k16, whose register layout is
// documented, so every thread knows the (row, token) of its two results and can apply the per-block
// weight and activation scales. One k16 step is exactly one Q6_K sub-block (half a 32-block).
// 128 threads = 2x2 warps over a 64x64 output tile; each warp owns 32 rows x 32 tokens (4x4 mma tiles).
// K advances 128 values (4 blocks of 32) per shared-memory stage. Rows are padded to 36 words so the
// fragment loads (8 rows x 4 words per warp) hit 32 different banks.
#define MMA_KB 4
#define MMA_LD 36
__global__ void matmul_mma(int fmt, const unsigned char* __restrict__ wq, const float* __restrict__ wd,
                           const signed char* __restrict__ xq, const float* __restrict__ xd, const int* __restrict__ xsum,
                           float* __restrict__ out, int rows, int cols, int n, int accumulate, int out_stride, int out_off) {
    __shared__ int ws[TILE * MMA_LD];
    __shared__ int xs[TILE * MMA_LD];
    __shared__ float wsa[TILE][MMA_KB], wsb[TILE][MMA_KB], wmn[TILE][MMA_KB], xds[TILE][MMA_KB], xms[TILE][MMA_KB];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int row0 = blockIdx.x * TILE, tok0 = blockIdx.y * TILE;
    int wr = (warp >> 1) * 32, wt = (warp & 1) * 32; // this warp's rows / tokens inside the tile
    int nb = cols >> 5, nsb = cols >> 8;
    int fr = lane >> 2, fk = lane & 3; // fragment row (or token) and word within the k16 step
    float acc[4][4][2];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) { acc[i][j][0] = 0.f; acc[i][j][1] = 0.f; }

    // Software pipeline: each thread fetches the next stage into registers (its global loads stay in
    // flight) while the warps run tensor-core math on the current stage in shared memory, then writes
    // it to shared memory. Per stage and thread (128 threads): 16 weight words, 16 activation words,
    // 2 (row, block) scale entries.
    int wreg[16], xreg[16];
    float sreg[2][5];
    auto fetch = [&](int kc) {
        int nk = min(MMA_KB, nb - kc);
        if (fmt == 1) {
            // Q4_K: one word holds the low nibbles of block 2p and the high nibbles of block 2p+1
            #pragma unroll
            for (int s = 0; s < 8; s++) {
                int idx = tid + 128 * s;
                int r = idx >> 4, pi = (idx >> 3) & 1, w = idx & 7;
                int gr = row0 + r, kk = 2 * pi, kb = kc + kk;
                int p = 0;
                if (kk < nk && gr < rows) {
                    const unsigned char* b = wq + ((size_t)gr * nsb + (kb >> 3)) * Q4K_BYTES;
                    p = *reinterpret_cast<const int*>(b + 16 + 32 * ((kb & 7) >> 1) + 4 * w);
                }
                wreg[2 * s] = p & 0x0F0F0F0F;
                wreg[2 * s + 1] = (kk + 1 < nk) ? ((p >> 4) & 0x0F0F0F0F) : 0;
            }
        } else {
            #pragma unroll
            for (int s = 0; s < 16; s++) {
                int idx = tid + 128 * s;
                int r = idx >> 5, wi = idx & 31, kk = wi >> 3, w = wi & 7;
                int gr = row0 + r, kb = kc + kk, k8 = kb & 7;
                int v = 0;
                if (kk < nk && gr < rows) {
                    if (fmt == 0) {
                        v = *reinterpret_cast<const int*>(wq + (size_t)gr * cols + kb * 32 + w * 4);
                    } else {
                        const unsigned char* b = wq + ((size_t)gr * nsb + (kb >> 3)) * Q6K_BYTES;
                        int half = k8 >> 2, g = k8 & 3;
                        int p = *reinterpret_cast<const int*>(b + 64 * half + (g & 1) * 32 + 4 * w);
                        int lo = (g >= 2) ? ((p >> 4) & 0x0F0F0F0F) : (p & 0x0F0F0F0F);
                        int h = *reinterpret_cast<const int*>(b + 128 + 32 * half + 4 * w);
                        v = __vsub4(lo | (((h >> (2 * g)) & 0x03030303) << 4), 0x20202020);
                    }
                }
                wreg[s] = v;
            }
        }
        #pragma unroll
        for (int s = 0; s < 16; s++) {
            int idx = tid + 128 * s;
            int r = idx >> 5, wi = idx & 31, kk = wi >> 3, w = wi & 7;
            int gt = tok0 + r;
            xreg[s] = (kk < nk && gt < n) ? *reinterpret_cast<const int*>(xq + (size_t)gt * cols + (kc + kk) * 32 + w * 4) : 0;
        }
        #pragma unroll
        for (int s = 0; s < 2; s++) {
            int idx = tid + 128 * s;
            int r = idx / MMA_KB, kk = idx % MMA_KB, kb = kc + kk, k8 = kb & 7;
            int gr = row0 + r, gt = tok0 + r;
            float sa = 0.f, sb = 0.f, mn = 0.f;
            if (kk < nk && gr < rows) {
                if (fmt == 0) {
                    sa = sb = wd[(size_t)gr * nb + kb];
                } else if (fmt == 1) {
                    const unsigned char* b = wq + ((size_t)gr * nsb + (kb >> 3)) * Q4K_BYTES;
                    int sc, m;
                    q4k_scale_min(k8, b + 4, &sc, &m);
                    float d, dmin;
                    q4k_dm(b, &d, &dmin);
                    sa = sb = d * sc;
                    mn = dmin * m;
                } else {
                    const unsigned char* b = wq + ((size_t)gr * nsb + (kb >> 3)) * Q6K_BYTES;
                    float d = wd[(size_t)gr * nsb + (kb >> 3)];
                    const signed char* sc = (const signed char*)(b + 192 + 8 * (k8 >> 2) + 2 * (k8 & 3));
                    sa = d * sc[0];
                    sb = d * sc[1];
                }
            }
            bool ok = kk < nk && gt < n;
            float dx = ok ? xd[(size_t)gt * nb + kb] : 0.f;
            sreg[s][0] = sa; sreg[s][1] = sb; sreg[s][2] = mn;
            sreg[s][3] = dx;
            sreg[s][4] = ok ? dx * (float)xsum[(size_t)gt * nb + kb] : 0.f;
        }
    };
    auto commit = [&]() {
        if (fmt == 1) {
            #pragma unroll
            for (int s = 0; s < 8; s++) {
                int idx = tid + 128 * s;
                int r = idx >> 4, kk = 2 * ((idx >> 3) & 1), w = idx & 7;
                ws[r * MMA_LD + kk * 8 + w] = wreg[2 * s];
                ws[r * MMA_LD + (kk + 1) * 8 + w] = wreg[2 * s + 1];
            }
        } else {
            #pragma unroll
            for (int s = 0; s < 16; s++) {
                int idx = tid + 128 * s;
                ws[(idx >> 5) * MMA_LD + (idx & 31)] = wreg[s];
            }
        }
        #pragma unroll
        for (int s = 0; s < 16; s++) {
            int idx = tid + 128 * s;
            xs[(idx >> 5) * MMA_LD + (idx & 31)] = xreg[s];
        }
        #pragma unroll
        for (int s = 0; s < 2; s++) {
            int idx = tid + 128 * s;
            int r = idx / MMA_KB, kk = idx % MMA_KB;
            wsa[r][kk] = sreg[s][0]; wsb[r][kk] = sreg[s][1]; wmn[r][kk] = sreg[s][2];
            xds[r][kk] = sreg[s][3]; xms[r][kk] = sreg[s][4];
        }
    };

    fetch(0);
    commit();
    __syncthreads();
    for (int kc = 0; kc < nb; kc += MMA_KB) {
        int nk = min(MMA_KB, nb - kc);
        bool more = kc + MMA_KB < nb;
        if (more) fetch(kc + MMA_KB); // issued now, consumed after the math below

        for (int kk = 0; kk < nk; kk++) {
            // this thread's scales for the block, in registers: 4 rows, 8 tokens
            float ra[4], rb[4], ts[4][2];
            #pragma unroll
            for (int i = 0; i < 4; i++) { ra[i] = wsa[wr + i * 8 + fr][kk]; rb[i] = wsb[wr + i * 8 + fr][kk]; }
            #pragma unroll
            for (int j = 0; j < 4; j++) { ts[j][0] = xds[wt + j * 8 + fk * 2][kk]; ts[j][1] = xds[wt + j * 8 + fk * 2 + 1][kk]; }
            int c[4][4][2];
            #pragma unroll
            for (int i = 0; i < 4; i++)
                #pragma unroll
                for (int j = 0; j < 4; j++) { c[i][j][0] = 0; c[i][j][1] = 0; }
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                int kw = kk * 8 + h * 4 + fk;
                int a[4], bq[4];
                #pragma unroll
                for (int i = 0; i < 4; i++) a[i] = ws[(wr + i * 8 + fr) * MMA_LD + kw];
                #pragma unroll
                for (int j = 0; j < 4; j++) bq[j] = xs[(wt + j * 8 + fr) * MMA_LD + kw];
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    #pragma unroll
                    for (int j = 0; j < 4; j++)
                        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                            : "+r"(c[i][j][0]), "+r"(c[i][j][1]) : "r"(a[i]), "r"(bq[j]));
                // Q6_K has a scale per 16 values (one k16 step): fold in each half separately.
                // Q8_0 / Q4_K have one scale per 32: integers keep accumulating, folded in once below.
                if (fmt == 2) {
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        float s = h ? rb[i] : ra[i];
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            acc[i][j][0] += (float)c[i][j][0] * (s * ts[j][0]);
                            acc[i][j][1] += (float)c[i][j][1] * (s * ts[j][1]);
                            c[i][j][0] = 0; c[i][j][1] = 0;
                        }
                    }
                }
            }
            if (fmt != 2) {
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    float mn = wmn[wr + i * 8 + fr][kk]; // Q4_K mins (0 for Q8_0)
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        int t = wt + j * 8 + fk * 2;
                        acc[i][j][0] += (float)c[i][j][0] * (ra[i] * ts[j][0]) - mn * xms[t][kk];
                        acc[i][j][1] += (float)c[i][j][1] * (ra[i] * ts[j][1]) - mn * xms[t + 1][kk];
                    }
                }
            }
        }
        __syncthreads(); // everyone done reading this stage
        if (more) {
            commit();
            __syncthreads();
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = row0 + wr + i * 8 + fr;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                int t = tok0 + wt + j * 8 + fk * 2 + e;
                if (r < rows && t < n) {
                    size_t idx = (size_t)t * out_stride + out_off + r;
                    out[idx] = accumulate ? out[idx] + acc[i][j][e] : acc[i][j][e];
                }
            }
        }
    }
}

// Activations for the K-quant prompt matmuls: int8 with one scale per QCH values (128 or 256, set by the
// compile options; one warp per chunk), plus each 32-block's sum split into a high part (sum >> 6) and the
// low 6 bits, so the Q4_K min term fits int8 tensor cores. Per chunk, QCH/64 sum words: the QCH/32 high
// bytes, then the QCH/32 low bytes.
#define QVPL (QCH / 32) // values per lane
__global__ void quantize_q8k(const float* __restrict__ x, signed char* __restrict__ q, float* __restrict__ d,
                             int* __restrict__ sums, int n_chunks) {
    int c = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (c >= n_chunks) return;
    const float4* src = reinterpret_cast<const float4*>(x + (size_t)c * QCH + lane * QVPL);
    float v[QVPL];
    #pragma unroll
    for (int i = 0; i < QVPL / 4; i++) { float4 f = src[i]; v[4 * i] = f.x; v[4 * i + 1] = f.y; v[4 * i + 2] = f.z; v[4 * i + 3] = f.w; }
    float amax = 0.f;
    #pragma unroll
    for (int i = 0; i < QVPL; i++) amax = fmaxf(amax, fabsf(v[i]));
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    float dd = amax / 127.f;
    float inv = dd != 0.f ? 1.f / dd : 0.f;
    int s = 0, packed[QVPL / 4];
    #pragma unroll
    for (int i = 0; i < QVPL / 4; i++) {
        int q0 = (int)roundf(v[4 * i] * inv), q1 = (int)roundf(v[4 * i + 1] * inv);
        int q2 = (int)roundf(v[4 * i + 2] * inv), q3 = (int)roundf(v[4 * i + 3] * inv);
        packed[i] = (q0 & 255) | ((q1 & 255) << 8) | ((q2 & 255) << 16) | ((unsigned)q3 << 24);
        s += q0 + q1 + q2 + q3;
    }
    int* dst = reinterpret_cast<int*>(q + (size_t)c * QCH + lane * QVPL);
    #pragma unroll
    for (int i = 0; i < QVPL / 4; i++) dst[i] = packed[i];
    for (int o = 1; o < 32 / QVPL; o <<= 1) s += __shfl_xor_sync(0xffffffff, s, o); // lanes of one 32-block
    if (lane == 0) d[c] = dd;
    if ((lane & (32 / QVPL - 1)) == 0) {
        signed char* bs = reinterpret_cast<signed char*>(sums + (size_t)c * (QCH / 64));
        int j = lane / (32 / QVPL);
        bs[j] = (signed char)(s >> 6);
        bs[QCH / 32 + j] = (signed char)(s & 63);
    }
}

}  // extern "C": the template below needs C++ linkage

// K-quant prompt matmul with an integer epilogue (activations from quantize_q8k). Within one activation
// scale (QCH values) everything stays integer: the 6-bit (Q4_K) or int8 (Q6_K) sub-block scales multiply
// the tensor-core sums with IMAD, the Q4_K mins go through one extra pair of mma (mins x the split
// activation sums), and floats are touched once per QCH values instead of once per 32 or 16.
// Same tiling as matmul_mma: 64x64 tile, 2x2 warps of 32x32, 128 K-values per shared-memory stage, so
// two stages per super-block. Per tile and 32-block the two k16 mma run back to back into a fresh pair
// of registers, which keeps the per-block integer sums out of the register file.
#define XBW (QCH / 64) // split-sum words per token and activation chunk
template <int FMT>
__device__ void mma_kquant(const unsigned char* __restrict__ wq, const float* __restrict__ wd, const signed char* __restrict__ xq,
                           const float* __restrict__ xd, const int* __restrict__ xb, float* __restrict__ out,
                           int rows, int cols, int n, int accumulate, int out_stride, int out_off) {
    __shared__ __align__(16) int ws[TILE * MMA_LD];
    __shared__ __align__(16) int xs[TILE * MMA_LD];
    __shared__ float wdm[TILE][2]; // d, and dmin for Q4_K
    __shared__ int wsc[TILE][2];   // this stage's sub-block scales as packed bytes (Q4_K: 4, Q6_K: 8)
    __shared__ int wmn[TILE][2];   // Q4_K: the super-block's 8 mins as packed bytes
    __shared__ float xds[TILE];    // activation scale of each token for the current QCH chunk
    __shared__ int xbs[TILE][4];   // its split 32-block sums (XBW words used)
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int row0 = blockIdx.x * TILE, tok0 = blockIdx.y * TILE;
    int wr = (warp >> 1) * 32, wt = (warp & 1) * 32;
    int nsb = cols >> 8, nst = 2 * nsb;
    int fr = lane >> 2, fk = lane & 3;
    const int bytes = FMT == 1 ? Q4K_BYTES : Q6K_BYTES;
    float acc[4][4][2];
    int iacc[4][4][2];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) { acc[i][j][0] = acc[i][j][1] = 0.f; iacc[i][j][0] = iacc[i][j][1] = 0; }

    // Register-staged pipeline as in matmul_mma. fetch() only issues loads into raw words; all the
    // unpacking happens in commit(), after the math, so no thread waits on its loads early.
    // Quant bytes and activations move in 16-byte pieces (128-bit loads and shared stores). Each group
    // of 8 lanes reads one cache line (activations: 8 pieces of one token) or one piece of 8 rows
    // (weights), and the 8 lanes of a shared store hit 32 different banks. Rows and tokens past the end
    // are clamped instead of tested; they only feed results that are never stored.
    int4 wraw[FMT == 1 ? 2 : 3], xreg[4];
    const signed char* xp[4];
    #pragma unroll
    for (int s = 0; s < 4; s++) xp[s] = xq + (size_t)min(tok0 + (tid >> 3) + 16 * s, n - 1) * cols + 16 * (tid & 7);
    // weights: Q4_K piece q (of 4 per row and stage) of rows wl + 32*s; Q6_K piece half wh of row wl
    int wl = FMT == 1 ? (tid & 7) + 8 * (tid >> 5) : (tid & 7) + 8 * (tid >> 4);
    int wqp = FMT == 1 ? (tid >> 3) & 3 : (tid >> 3) & 1;
    const unsigned char* wp[FMT == 1 ? 2 : 1];
    #pragma unroll
    for (int s = 0; s < (FMT == 1 ? 2 : 1); s++) wp[s] = wq + (size_t)min(row0 + wl + 32 * s, rows - 1) * nsb * bytes;
    // metadata: two threads per row and per token
    int lr = tid >> 1, hh = tid & 1;
    int grc = min(row0 + lr, rows - 1), gtc = min(tok0 + lr, n - 1);
    const unsigned char* mrow = wq + (size_t)grc * nsb * bytes;
    const float* wdrow = wd + (size_t)grc * nsb; // Q6_K
    float xf;
    unsigned int wm, ma0, ma1; // wm: Q4_K d|dmin as two f16, Q6_K the f32 d's bits
    int xi0, xi1;
    // Two halves, issued at different points of the math so the load queue doesn't overflow:
    // the weights and their metadata (half 0), the activations and theirs (half 1).
    auto fetch = [&](int st, int half) {
        int sb = st >> 1, hs = st & 1;
        if (half == 1) {
            #pragma unroll
            for (int s = 0; s < 4; s++) xreg[s] = *reinterpret_cast<const int4*>(xp[s] + st * 128);
            size_t c = (size_t)gtc * (cols / QCH) + st * 128 / QCH;
            xf = xd[c];
            xi0 = xb[c * XBW + hh * (XBW / 2)];
            xi1 = XBW == 4 ? xb[c * XBW + hh * (XBW / 2) + 1] : 0;
            return;
        }
        if (FMT == 1) {
            // nibble bytes 64*hs .. +63 = [blocks 0 (low) / 1 (high): 32 bytes] [blocks 2 / 3: 32 bytes]
            #pragma unroll
            for (int s = 0; s < 2; s++) wraw[s] = *reinterpret_cast<const int4*>(wp[s] + (size_t)sb * bytes + 16 + 64 * hs + 16 * wqp);
        } else {
            // half wqp (16 bytes) of the low-nibble words of blocks 0/2, of blocks 1/3, and of the 2-bit highs
            const unsigned char* b = wp[0] + (size_t)sb * bytes;
            wraw[0] = *reinterpret_cast<const int4*>(b + 64 * hs + 16 * wqp);
            wraw[FMT - 1] = *reinterpret_cast<const int4*>(b + 64 * hs + 32 + 16 * wqp);
            wraw[2 * (FMT - 1)] = *reinterpret_cast<const int4*>(b + 128 + 32 * hs + 16 * wqp);
        }
        const unsigned char* b = mrow + (size_t)sb * bytes;
        if (FMT == 1) {
            wm = *reinterpret_cast<const unsigned int*>(b);                     // d, dmin
            ma0 = *reinterpret_cast<const unsigned int*>(b + 4 + 4 * hh);       // scales / mins of sub-blocks 0..3
            ma1 = *reinterpret_cast<const unsigned int*>(b + 12);               // low 4 bits of both for 4..7
        } else {
            wm = __float_as_uint(wdrow[sb]);
            ma0 = *reinterpret_cast<const unsigned int*>(b + 192 + 8 * hs + 4 * hh);
        }
    };
    auto commit = [&](int st) {
        int hs = st & 1;
        float wf = FMT == 1 ? h2f(hh ? wm >> 16 : wm & 0xFFFF) : __uint_as_float(wm);
        if (FMT == 1) {
            // piece q = words 4*(q&1) .. +3 of the 32 bytes of blocks 2*(q>>1) (low) and 2*(q>>1)+1 (high)
            int kb = 2 * (wqp >> 1), h = wqp & 1;
            #pragma unroll
            for (int s = 0; s < 2; s++) {
                int4 p = wraw[s];
                int* dst = ws + (wl + 32 * s) * MMA_LD + kb * 8 + h * 4;
                *reinterpret_cast<int4*>(dst) = make_int4(p.x & 0x0F0F0F0F, p.y & 0x0F0F0F0F, p.z & 0x0F0F0F0F, p.w & 0x0F0F0F0F);
                *reinterpret_cast<int4*>(dst + 8) = make_int4((p.x >> 4) & 0x0F0F0F0F, (p.y >> 4) & 0x0F0F0F0F, (p.z >> 4) & 0x0F0F0F0F, (p.w >> 4) & 0x0F0F0F0F);
            }
        } else {
            int l0[4] = {wraw[0].x, wraw[0].y, wraw[0].z, wraw[0].w};
            int l1[4] = {wraw[FMT - 1].x, wraw[FMT - 1].y, wraw[FMT - 1].z, wraw[FMT - 1].w};
            int hb[4] = {wraw[2 * (FMT - 1)].x, wraw[2 * (FMT - 1)].y, wraw[2 * (FMT - 1)].z, wraw[2 * (FMT - 1)].w};
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                int v[4];
                #pragma unroll
                for (int f = 0; f < 4; f++) {
                    int p = (g & 1) ? l1[f] : l0[f];
                    int lo = (g >= 2) ? ((p >> 4) & 0x0F0F0F0F) : (p & 0x0F0F0F0F);
                    v[f] = __vsub4(lo | (((hb[f] >> (2 * g)) & 0x03030303) << 4), 0x20202020); // bytewise -32
                }
                *reinterpret_cast<int4*>(ws + wl * MMA_LD + g * 8 + wqp * 4) = make_int4(v[0], v[1], v[2], v[3]);
            }
        }
        #pragma unroll
        for (int s = 0; s < 4; s++) *reinterpret_cast<int4*>(xs + ((tid >> 3) + 16 * s) * MMA_LD + 4 * (tid & 7)) = xreg[s];
        if (FMT == 1) {
            // 6-bit fields: sub-blocks 0..3 in the low 6 bits of ma0's bytes; 4..7 from ma1's nibbles + ma0's top 2 bits
            unsigned int hi = ((ma1 >> (4 * hh)) & 0x0F0F0F0F) | ((ma0 >> 2) & 0x30303030);
            wdm[lr][hh] = wf;
            if (hh == 0) wsc[lr][0] = (int)(hs ? hi : ma0 & 0x3F3F3F3F);
            else { wmn[lr][0] = (int)(ma0 & 0x3F3F3F3F); wmn[lr][1] = (int)hi; }
        } else {
            if (hh == 0) wdm[lr][0] = wf;
            wsc[lr][hh] = (int)ma0;
        }
        if (hh == 0) xds[lr] = xf;
        xbs[lr][hh * (XBW / 2)] = xi0;
        if (XBW == 4) xbs[lr][2 * hh + 1] = xi1;
    };

    // shared-memory byte addresses of this lane's ldmatrix rows (rows 36 words apart: conflict-free)
    unsigned lda = (unsigned)__cvta_generic_to_shared(ws + (wr + (lane >> 3) * 8 + (lane & 7)) * MMA_LD);
    unsigned ldb = (unsigned)__cvta_generic_to_shared(xs + (wt + (lane >> 3) * 8 + (lane & 7)) * MMA_LD);

    // ldmatrix.x4: four 8x8 b16 matrices (= 8 rows x 16 int8) per instruction, already in the m8n8k16
    // fragment layout; lane l gives the address of row l&7 of matrix l>>3. Block kb's fragments are
    // loaded while block kb-1 is still in the tensor cores.
    int fa[2][4][2], fb[2][4][2];
    auto frags = [&](int kb, int buf) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(fa[buf][0][h]), "=r"(fa[buf][1][h]), "=r"(fa[buf][2][h]), "=r"(fa[buf][3][h]) : "r"(lda + (kb * 8 + h * 4) * 4));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(fb[buf][0][h]), "=r"(fb[buf][1][h]), "=r"(fb[buf][2][h]), "=r"(fb[buf][3][h]) : "r"(ldb + (kb * 8 + h * 4) * 4));
        }
    };

    fetch(0, 0);
    fetch(0, 1);
    commit(0);
    __syncthreads();
    for (int st = 0; st < nst; st++) {
        bool more = st + 1 < nst;
        frags(0, 0);
        if (more) fetch(st + 1, 0); // issued now, consumed after the math below

        int scw[4][FMT == 1 ? 1 : 2];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            scw[i][0] = wsc[wr + i * 8 + fr][0];
            if (FMT == 2) scw[i][FMT - 1] = wsc[wr + i * 8 + fr][1];
        }
        #pragma unroll
        for (int kb = 0; kb < 4; kb++) {
            if (kb < 3) frags(kb + 1, (kb + 1) & 1);
            if (kb == 1 && more) fetch(st + 1, 1);
            int (&a)[4][2] = fa[kb & 1];
            int (&bq)[4][2] = fb[kb & 1];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    if (FMT == 1) {
                        // Q4_K: one scale per 32 values
                        int sc = (scw[i][0] >> (8 * kb)) & 0xFF;
                        int c0 = 0, c1 = 0;
                        #pragma unroll
                        for (int h = 0; h < 2; h++)
                            asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                                : "+r"(c0), "+r"(c1) : "r"(a[i][h]), "r"(bq[j][h]));
                        iacc[i][j][0] += sc * c0;
                        iacc[i][j][1] += sc * c1;
                    } else {
                        // Q6_K: one signed scale per 16 values, i.e. per k16 step
                        #pragma unroll
                        for (int h = 0; h < 2; h++) {
                            int sc = (int)(signed char)(scw[i][(FMT - 1) * (kb >> 1)] >> (8 * ((2 * kb + h) & 3)));
                            int c0 = 0, c1 = 0;
                            asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                                : "+r"(c0), "+r"(c1) : "r"(a[i][h]), "r"(bq[j][h]));
                            iacc[i][j][0] += sc * c0;
                            iacc[i][j][1] += sc * c1;
                        }
                    }
                }
            }
        }
        if ((st + 1) % (QCH / 128) == 0) {
            // activation chunk done: one float fold per result, plus the Q4_K mins of its sub-blocks
            const int nf = QCH / 128; // fragment words holding those mins / sums (4 per word)
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int r = wr + i * 8 + fr;
                float d = wdm[r][0], dm = FMT == 1 ? wdm[r][1] : 0.f;
                int ma = (FMT == 1 && fk < nf) ? wmn[r][(nf == 1 ? (st & 1) : 0) + fk] : 0; // A fragment: mins, zeros above
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int t = wt + j * 8 + fk * 2;
                    float D0 = xds[t], D1 = xds[t + 1];
                    int m0 = 0, m1 = 0;
                    if (FMT == 1) {
                        int tb = wt + j * 8 + fr; // B fragment: column = token
                        int bh = fk < nf ? xbs[tb][fk] : 0, bl = fk < nf ? xbs[tb][nf + fk] : 0;
                        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                            : "+r"(m0), "+r"(m1) : "r"(ma), "r"(bh));
                        m0 *= 64; m1 *= 64;
                        asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                            : "+r"(m0), "+r"(m1) : "r"(ma), "r"(bl));
                    }
                    acc[i][j][0] += D0 * (d * (float)iacc[i][j][0] - dm * (float)m0);
                    acc[i][j][1] += D1 * (d * (float)iacc[i][j][1] - dm * (float)m1);
                    iacc[i][j][0] = iacc[i][j][1] = 0;
                }
            }
        }
        __syncthreads(); // everyone done reading this stage
        if (more) {
            commit(st + 1);
            __syncthreads();
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int r = row0 + wr + i * 8 + fr;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                int t = tok0 + wt + j * 8 + fk * 2 + e;
                if (r < rows && t < n) {
                    size_t idx = (size_t)t * out_stride + out_off + r;
                    out[idx] = accumulate ? out[idx] + acc[i][j][e] : acc[i][j][e];
                }
            }
        }
    }
}

extern "C" {

#define MMA_K_ARGS const unsigned char* __restrict__ wq, const float* __restrict__ wd, const signed char* __restrict__ xq, const float* __restrict__ xd, \
                   const int* __restrict__ xb, float* __restrict__ out, int rows, int cols, int n, int accumulate, int out_stride, int out_off
__global__ void __launch_bounds__(128) matmul_mma_q4k(MMA_K_ARGS) { mma_kquant<1>(wq, wd, xq, xd, xb, out, rows, cols, n, accumulate, out_stride, out_off); }
__global__ void __launch_bounds__(128) matmul_mma_q6k(MMA_K_ARGS) { mma_kquant<2>(wq, wd, xq, xd, xb, out, rows, cols, n, accumulate, out_stride, out_off); }

// Decode: h = silu(gate) * up straight to int8 activations for the down projection (ffn a multiple of 32),
// one token per grid row.
__global__ void silu_q8(const float* __restrict__ gu, signed char* __restrict__ xq, float* __restrict__ xd,
                        int* __restrict__ xsum, int ffn) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, nv = ffn >> 2;
    size_t r = blockIdx.y;
    gu += r * 2 * ffn;
    xq += r * ffn;
    xd += r * (ffn >> 5);
    xsum += r * (ffn >> 5);
    bool ok = i < nv;
    float4 h = make_float4(0.f, 0.f, 0.f, 0.f);
    if (ok) {
        float4 g = reinterpret_cast<const float4*>(gu)[i], u = reinterpret_cast<const float4*>(gu + ffn)[i];
        h = make_float4(g.x / (1.f + expf(-g.x)) * u.x, g.y / (1.f + expf(-g.y)) * u.y,
                        g.z / (1.f + expf(-g.z)) * u.z, g.w / (1.f + expf(-g.w)) * u.w);
    }
    quant8x4(h, ok, i, xq, xd, xsum);
}

// Index of each row's largest value, as the greedy sampler picks it: the first maximum, NaN never wins,
// 0 when nothing beats -inf. One block per row.
__global__ void argmax_rows(const float* __restrict__ x, int cols, int* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    const float* r = x + (size_t)blockIdx.x * cols;
    float bv = NEG_INF;
    int bi = 0x7FFFFFFF;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = r[i];
        if (v > bv) { bv = v; bi = i; }
    }
    // better: larger value, or the same value at a lower index
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, bv, o);
        int oi = __shfl_xor_sync(0xffffffff, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { sv[wid] = bv; si[wid] = bi; }
    __syncthreads();
    if (wid == 0) {
        int nw = blockDim.x >> 5;
        bv = lane < nw ? sv[lane] : NEG_INF;
        bi = lane < nw ? si[lane] : 0x7FFFFFFF;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, bv, o);
            int oi = __shfl_xor_sync(0xffffffff, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) out[blockIdx.x] = bi == 0x7FFFFFFF ? 0 : bi;
    }
}

// gu rows are [gate | up]; h = silu(gate) * up
__global__ void silu_mul(const float* __restrict__ gu, float* __restrict__ h, int ffn, int n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * ffn) return;
    size_t b = i / ffn, j = i % ffn;
    float g = gu[b * 2 * ffn + j], u = gu[b * 2 * ffn + ffn + j];
    h[i] = g / (1.f + expf(-g)) * u;
}

// ---- mixture of experts, prompt batches ----

// Router logits: out[t * ne + e] = dot(xb[t], R[e]), f32 throughout (the routing must not depend on
// rounding). One block per token: its row goes to shared memory, each warp takes experts in turn.
__global__ void router_logits(const float* __restrict__ xb, const float* __restrict__ R, float* __restrict__ out, int dim, int ne) {
    extern __shared__ float row[];
    const float* x = xb + (size_t)blockIdx.x * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) row[d] = x[d];
    __syncthreads();
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    for (int e = warp; e < ne; e += nwarps) {
        const float* r = R + (size_t)e * dim;
        float acc = 0.f;
        for (int d = lane; d < dim; d += 32) acc += row[d] * r[d];
        acc = warp_sum(acc);
        if (lane == 0) out[(size_t)blockIdx.x * ne + e] = acc;
    }
}

// one token's router logits, a warp per expert: out[e] = R[e] . x (dim a multiple of 4)
__global__ void router_logits1(const float* __restrict__ x, const float* __restrict__ R, float* __restrict__ out, int dim, int ne) {
    int e = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (e >= ne) return;
    const float4* r = reinterpret_cast<const float4*>(R + (size_t)e * dim);
    const float4* v = reinterpret_cast<const float4*>(x);
    float acc = 0.f;
    for (int d = lane; d < dim / 4; d += 32) {
        float4 a = r[d], b = v[d];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    acc = warp_sum(acc);
    if (lane == 0) out[e] = acc;
}

// out[i] = xb[idx[i]] for the m rows an expert takes (one block per row)
__global__ void moe_gather(const float* __restrict__ xb, const int* __restrict__ idx, float* __restrict__ out, int dim) {
    const float4* s = reinterpret_cast<const float4*>(xb + (size_t)idx[blockIdx.x] * dim);
    float4* o = reinterpret_cast<float4*>(out + (size_t)blockIdx.x * dim);
    for (int d = threadIdx.x; d < dim / 4; d += blockDim.x) o[d] = s[d];
}

// x[idx[i]] += w[i] * y[i]; an expert takes each token at most once, so the rows never collide
__global__ void moe_scatter(float* __restrict__ x, const float* __restrict__ y, const int* __restrict__ idx, const float* __restrict__ w, int dim) {
    float* o = x + (size_t)idx[blockIdx.x] * dim;
    const float* s = y + (size_t)blockIdx.x * dim;
    float wi = w[blockIdx.x];
    for (int d = threadIdx.x; d < dim; d += blockDim.x) o[d] += wi * s[d];
}

// GPU Q3_K (112-byte blocks) to Q8: int8 per value and an f32 scale per 32 (requantised: a 32-value
// block spans two of the 16-value scales). One warp per 32 values.
__global__ void q3k_to_q8(const unsigned char* __restrict__ w, signed char* __restrict__ q, float* __restrict__ d, int blocks32) {
    int b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b >= blocks32) return;
    int sb = b >> 3, k = b & 7, half = k >> 2, g = k & 3, l = lane; // value l of group g in half `half`
    const unsigned char* blk = w + (size_t)sb * Q3K_GPU_BYTES;
    const unsigned* hdr = reinterpret_cast<const unsigned*>(blk);
    int qv = ((blk[48 + 32 * half + l] >> (2 * g)) & 3) - (((blk[16 + l] >> (4 * half + g)) & 1) ? 0 : 4);
    float v = h2f(hdr[3] & 0xFFFF) * (float)q3k_scale(8 * half + 2 * g + (l >> 4), hdr[0], hdr[1], hdr[2]) * (float)qv;
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    float dd = amax / 127.f;
    q[(size_t)b * 32 + l] = (signed char)__float2int_rn(dd != 0.f ? v / dd : 0.f);
    if (lane == 0) d[b] = dd;
}

// GGUF Q5_K (176-byte blocks: f16 d, f16 dmin, 12 scale bytes, 32 high-bit bytes, 128 nibble bytes) to
// Q8 as above. 32-value block k of a super-block is sub-block k (scale and min from the 12 bytes).
__global__ void q5k_to_q8(const unsigned char* __restrict__ w, signed char* __restrict__ q, float* __restrict__ d, int blocks32) {
    int b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b >= blocks32) return;
    int sb = b >> 3, k = b & 7, l = lane;
    const unsigned char* blk = w + (size_t)sb * 176;
    int sc, m;
    q4k_scale_min(k, blk + 4, &sc, &m);
    int lo = blk[48 + 32 * (k >> 1) + l];
    lo = (k & 1) ? (lo >> 4) : (lo & 15);
    int qv = lo | (((blk[16 + l] >> k) & 1) << 4);
    unsigned dm = *reinterpret_cast<const unsigned*>(blk);
    float v = h2f(dm & 0xFFFF) * sc * qv - h2f(dm >> 16) * m;
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    float dd = amax / 127.f;
    q[(size_t)b * 32 + l] = (signed char)__float2int_rn(dd != 0.f ? v / dd : 0.f);
    if (lane == 0) d[b] = dd;
}

// x[i] += y[i] (the residual add of a feed-forward computed elsewhere, e.g. experts on the CPU)
__global__ void add_rows(float* __restrict__ x, const float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// ---- multi-head latent attention (deepseek2, e.g. GLM-4.7-Flash) ----
// A token's cache row is kl + rd f16 values: the RMS-normed compressed key-value vector, then the RoPE key
// every head shares. The values are the compressed vector itself.

// one block per row: RMSNorm of `cols` values read with row stride xs, written with row stride os
__global__ void rmsnorm_rows(const float* __restrict__ x, int xs, const float* __restrict__ w, float* __restrict__ out,
                             int os, int cols, float eps) {
    __shared__ float sh[32];
    const float* xr = x + (size_t)blockIdx.x * xs;
    float* o = out + (size_t)blockIdx.x * os;
    float ss = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, sh);
    float scale = rsqrtf(ss / cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) o[i] = xr[i] * scale * w[i];
}

// One block per token: its compressed vector (normed) and RoPE key (rotated, adjacent pairs) into the cache
// row of its position. kva: rows of `xs` floats, the vector and key starting at `off`.
__global__ void mla_kv(const float* __restrict__ kva, int xs, int off, const float* __restrict__ w,
                       unsigned short* __restrict__ cache, int pos0, const int* __restrict__ posp,
                       int kl, int rd, float base, float eps) {
    __shared__ float sh[32];
    if (posp) pos0 = *posp;
    int b = blockIdx.x, pos = pos0 + b;
    const float* row = kva + (size_t)b * xs + off;
    unsigned short* dst = cache + (size_t)pos * (kl + rd);
    float ss = 0.f;
    for (int i = threadIdx.x; i < kl; i += blockDim.x) ss += row[i] * row[i];
    ss = block_sum(ss, sh);
    float scale = rsqrtf(ss / kl + eps);
    for (int i = threadIdx.x; i < kl; i += blockDim.x) dst[i] = f2h(row[i] * scale * w[i]);
    for (int i = threadIdx.x; i < rd / 2; i += blockDim.x) {
        float s, c;
        sincosf((float)pos * powf(base, -2.0f * (float)i / (float)rd), &s, &c);
        float a = row[kl + 2 * i], bb = row[kl + 2 * i + 1];
        dst[kl + 2 * i] = f2h(a * c - bb * s);
        dst[kl + 2 * i + 1] = f2h(a * s + bb * c);
    }
}

// Grid (tokens, heads): the RoPE part of each head's query (the last rd of its hd values in q), rotated,
// to the end of the head's row in qa (kl + rd values).
__global__ void mla_q_rope(const float* __restrict__ q, float* __restrict__ qa, int pos0, const int* __restrict__ posp,
                           int hd, int kl, int rd, float base) {
    if (posp) pos0 = *posp;
    int b = blockIdx.x, h = blockIdx.y, H = gridDim.y, pos = pos0 + b;
    const float* src = q + ((size_t)b * H + h) * hd + (hd - rd);
    float* dst = qa + ((size_t)b * H + h) * (kl + rd) + kl;
    for (int i = threadIdx.x; i < rd / 2; i += blockDim.x) {
        float s, c;
        sincosf((float)pos * powf(base, -2.0f * (float)i / (float)rd), &s, &c);
        float a = src[2 * i], bb = src[2 * i + 1];
        dst[2 * i] = a * c - bb * s;
        dst[2 * i + 1] = a * s + bb * c;
    }
}

// Per-head projection with f16 weights: y[t][h][r] = W[h][r] . x[t][h] (`rows` rows of `cols` values per
// head). Head h of token t is read at x + t * xs + h * xh and written at y + t * ys + h * yh. Grid
// (ceil(n / HMV_T), heads, row slices), 256 threads, HMV_T * cols floats of dynamic shared memory: a block
// reads its slice of the head's weights once for HMV_T tokens, one warp per row. Row slices spread a
// single token over the GPU.
#define HMV_T 8
__global__ void head_matvec(const unsigned short* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,
                            int rows, int cols, int n, int xs, int xh, int ys, int yh) {
    extern __shared__ float xin[];
    int h = blockIdx.y, t0 = blockIdx.x * HMV_T, nt = min(HMV_T, n - t0);
    for (int i = threadIdx.x; i < HMV_T * cols; i += blockDim.x) {
        int t = i / cols;
        xin[i] = t < nt ? x[(size_t)(t0 + t) * xs + (size_t)h * xh + i % cols] : 0.f;
    }
    __syncthreads();
    const unsigned short* wh = w + (size_t)h * rows * cols;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int per = (rows + gridDim.z - 1) / gridDim.z, r1 = min(rows, (int)(blockIdx.z + 1) * per);
    for (int r = blockIdx.z * per + warp; r < r1; r += blockDim.x >> 5) {
        const unsigned short* wr = wh + (size_t)r * cols;
        float acc[HMV_T];
#pragma unroll
        for (int t = 0; t < HMV_T; t++) acc[t] = 0.f;
        for (int c = lane; c < cols; c += 32) {
            float wv = h2f(wr[c]);
#pragma unroll
            for (int t = 0; t < HMV_T; t++) acc[t] += wv * xin[t * cols + c];
        }
#pragma unroll
        for (int t = 0; t < HMV_T; t++) {
            float v = warp_sum(acc[t]);
            if (lane == 0 && t < nt) y[(size_t)(t0 + t) * ys + (size_t)h * yh + r] = v;
        }
    }
}

// Latent attention. Grid (tokens, nsplit), 256 threads; token b sees positions 0..=pos0 + b, and block
// (b, s) takes the s-th of nsplit equal slices of them for all H heads at once, so each cache row is read
// once. Scores in tiles of MLA_T positions (a warp per position, every head), an online softmax per head,
// then each thread accumulates 2 of the kl value dims (kl == 2 * blockDim) for every head. nsplit == 1
// writes normalised outputs to out[b][h][kl]; otherwise the unnormalised sums, then each head's max and
// total, to part for mla_combine. Dynamic shared memory: (H * (kl + rd) + H * MLA_T + 3 * H) floats.
#define MLA_T 32
#define MLA_MAX_H 32
#define MLA_MAX_KW 24
__global__ void __launch_bounds__(256) mla_attention(const float* __restrict__ qa, const unsigned short* __restrict__ cache,
                                                     float* __restrict__ part, float* __restrict__ out, int pos0,
                                                     const int* __restrict__ posp, int H, int kl, int rd, float scale, int nsplit) {
    extern __shared__ float smem[];
    if (posp) pos0 = *posp;
    int b = blockIdx.x, sp = blockIdx.y, kd = kl + rd;
    int span = pos0 + b + 1, per = (span + nsplit - 1) / nsplit;
    int start = min(span, sp * per), end = min(span, start + per);
    float* sq = smem;               // H x kd: the queries
    float* ss = sq + H * kd;        // H x MLA_T: scores, then weights
    float* sm = ss + H * MLA_T;     // H: running max
    float* sl = sm + H;             // H: running total
    float* sc = sl + H;             // H: this tile's rescale
    const float* qb = qa + (size_t)b * H * kd;
    for (int i = threadIdx.x; i < H * kd; i += blockDim.x) sq[i] = qb[i];
    if (threadIdx.x < H) { sm[threadIdx.x] = NEG_INF; sl[threadIdx.x] = 0.f; }
    float acc[MLA_MAX_H][2];
#pragma unroll
    for (int h = 0; h < MLA_MAX_H; h++) acc[h][0] = acc[h][1] = 0.f;
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, kw = kd / 32;
    for (int t0 = start; t0 < end; t0 += MLA_T) {
        // scores of this tile: a warp per position
        for (int t = warp; t < MLA_T; t += 8) {
            int pos = t0 + t;
            float kv[MLA_MAX_KW];
            if (pos < end) {
                const unsigned short* kr = cache + (size_t)pos * kd;
#pragma unroll
                for (int i = 0; i < MLA_MAX_KW; i++) kv[i] = i < kw ? h2f(kr[lane + 32 * i]) : 0.f;
            }
            for (int h = 0; h < H; h++) {
                float d = 0.f;
                if (pos < end) {
                    const float* qh = sq + h * kd + lane;
#pragma unroll
                    for (int i = 0; i < MLA_MAX_KW; i++) if (i < kw) d += qh[32 * i] * kv[i];
                }
                d = warp_sum(d);
                if (lane == 0) ss[h * MLA_T + t] = pos < end ? d * scale : NEG_INF;
            }
        }
        __syncthreads();
        // online softmax, a thread per head
        if (threadIdx.x < H) {
            int h = threadIdx.x;
            float* row = ss + h * MLA_T;
            float m = sm[h];
            for (int t = 0; t < MLA_T; t++) m = fmaxf(m, row[t]);
            float c = __expf(sm[h] - m), total = 0.f;
            for (int t = 0; t < MLA_T; t++) { float e = __expf(row[t] - m); row[t] = e; total += e; }
            sl[h] = sl[h] * c + total;
            sm[h] = m;
            sc[h] = c;
        }
        __syncthreads();
        // values: this thread's 2 dims of every head
        int d0 = threadIdx.x, d1 = threadIdx.x + blockDim.x;
#pragma unroll
        for (int h = 0; h < MLA_MAX_H; h++) if (h < H) { acc[h][0] *= sc[h]; acc[h][1] *= sc[h]; }
        int nt = min(MLA_T, end - t0);
        for (int t = 0; t < nt; t++) {
            const unsigned short* vr = cache + (size_t)(t0 + t) * kd;
            float v0 = h2f(vr[d0]), v1 = h2f(vr[d1]);
#pragma unroll
            for (int h = 0; h < MLA_MAX_H; h++) if (h < H) {
                float p = ss[h * MLA_T + t];
                acc[h][0] += p * v0;
                acc[h][1] += p * v1;
            }
        }
        __syncthreads();
    }
    int d0 = threadIdx.x, d1 = threadIdx.x + blockDim.x;
    if (nsplit == 1) {
#pragma unroll
        for (int h = 0; h < MLA_MAX_H; h++) if (h < H) {
            float inv = sl[h] > 0.f ? 1.f / sl[h] : 0.f;
            float* o = out + ((size_t)b * H + h) * kl;
            o[d0] = acc[h][0] * inv;
            o[d1] = acc[h][1] * inv;
        }
        return;
    }
    float* pb = part + ((size_t)b * nsplit + sp) * H * (kl + 2);
#pragma unroll
    for (int h = 0; h < MLA_MAX_H; h++) if (h < H) {
        pb[h * kl + d0] = acc[h][0];
        pb[h * kl + d1] = acc[h][1];
    }
    if (threadIdx.x < H) {
        pb[H * kl + 2 * threadIdx.x] = sm[threadIdx.x];
        pb[H * kl + 2 * threadIdx.x + 1] = sl[threadIdx.x];
    }
}

// Grid (tokens, heads): the nsplit slices of mla_attention merged into normalised outputs.
__global__ void mla_combine(const float* __restrict__ part, float* __restrict__ out, int H, int kl, int nsplit) {
    int b = blockIdx.x, h = blockIdx.y;
    const float* pb = part + (size_t)b * nsplit * H * (kl + 2);
    size_t stride = (size_t)H * (kl + 2);
    float m = NEG_INF;
    for (int s = 0; s < nsplit; s++) m = fmaxf(m, pb[s * stride + H * kl + 2 * h]);
    float total = 0.f;
    for (int s = 0; s < nsplit; s++) {
        float ms = pb[s * stride + H * kl + 2 * h];
        if (ms > NEG_INF) total += pb[s * stride + H * kl + 2 * h + 1] * __expf(ms - m);
    }
    float inv = total > 0.f ? 1.f / total : 0.f;
    for (int d = threadIdx.x; d < kl; d += blockDim.x) {
        float v = 0.f;
        for (int s = 0; s < nsplit; s++) {
            float ms = pb[s * stride + H * kl + 2 * h];
            if (ms > NEG_INF) v += pb[s * stride + h * kl + d] * __expf(ms - m);
        }
        out[((size_t)b * H + h) * kl + d] = v * inv;
    }
}

// ---- Qwen3-Next: gated attention with 256-wide heads, Gated DeltaNet, the gated shared expert

// The attn_q rows hold [q | gate] per head, then k and v (src, src_stride floats apart): q, k, v to the packed
// [q | k | v] rows of qkv (as qk_norm_rope reads them) and the gates aside. grid (tokens, n_heads + 2 * n_kv).
__global__ void next_split_qgate(const float* __restrict__ src, int src_stride, float* __restrict__ qkv, float* __restrict__ gate,
                                 int n_heads, int n_kv, int hd) {
    int b = blockIdx.x, h = blockIdx.y;
    int q_dim = n_heads * hd, kv_dim = n_kv * hd;
    const float* s = src + (size_t)b * src_stride;
    float* o = qkv + (size_t)b * (q_dim + 2 * kv_dim);
    if (h < n_heads) {
        for (int d = threadIdx.x; d < hd; d += blockDim.x) {
            o[h * hd + d] = s[2 * h * hd + d];
            gate[(size_t)b * q_dim + h * hd + d] = s[2 * h * hd + hd + d];
        }
    } else {
        int j = h - n_heads; // the k heads, then the v heads
        for (int d = threadIdx.x; d < hd; d += blockDim.x) o[q_dim + j * hd + d] = s[2 * q_dim + j * hd + d];
    }
}

// Causal attention over the cache for heads wider than ATT_MAX_HD (a multiple of 128, up to 256), any
// number of tokens: a block per (token, query head), whose 8 warps take the positions in turn, each lane
// holding hd / 32 of the dimensions, with a running softmax per warp merged at the end. out: n x q_dim.
#define WIDE_WARPS 8
__global__ void attention_wide(const float* __restrict__ qkv, const kv_t* __restrict__ kc, const kv_t* __restrict__ vc,
                               const float* __restrict__ ks, const float* __restrict__ vs, float* __restrict__ out,
                               int pos0, int n_heads, int n_kv, int hd, float scale) {
    __shared__ float wm[WIDE_WARPS], wl[WIDE_WARPS], wo[WIDE_WARPS][256];
    int b = blockIdx.x, h = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int q_dim = n_heads * hd, kv_dim = n_kv * hd, kvh = h / (n_heads / n_kv), per = hd >> 5;
    const float* q = qkv + (size_t)b * (q_dim + 2 * kv_dim) + h * hd + lane * per;
    float qr[8], acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) { qr[i] = i < per ? q[i] * scale : 0.f; acc[i] = 0.f; }
    float m = NEG_INF, l = 0.f;
    int span = pos0 + b + 1;
    for (int t = warp; t < span; t += WIDE_WARPS) {
        const kv_t* kt = kc + (size_t)t * kv_dim + kvh * hd + lane * per;
        float s = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; i += 4) {
            if (i < per) { float4 k4 = kv_load4(kt + i); s += qr[i] * k4.x + qr[i + 1] * k4.y + qr[i + 2] * k4.z + qr[i + 3] * k4.w; }
        }
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        s *= KV_SCALE(ks, t, n_kv, kvh);
        float mn = fmaxf(m, s), c = __expf(m - mn), p = __expf(s - mn);
        l = l * c + p;
        const kv_t* vt = vc + (size_t)t * kv_dim + kvh * hd + lane * per;
        float pv = p * KV_SCALE(vs, t, n_kv, kvh);
        #pragma unroll
        for (int i = 0; i < 8; i += 4) {
            if (i < per) {
                float4 v4 = kv_load4(vt + i);
                acc[i] = acc[i] * c + pv * v4.x; acc[i + 1] = acc[i + 1] * c + pv * v4.y;
                acc[i + 2] = acc[i + 2] * c + pv * v4.z; acc[i + 3] = acc[i + 3] * c + pv * v4.w;
            }
        }
        m = mn;
    }
    if (lane == 0) { wm[warp] = m; wl[warp] = l; }
    for (int i = 0; i < per; i++) wo[warp][lane * per + i] = acc[i];
    __syncthreads();
    float mx = NEG_INF;
    for (int w = 0; w < WIDE_WARPS; w++) mx = fmaxf(mx, wm[w]);
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float v = 0.f, tot = 0.f;
        for (int w = 0; w < WIDE_WARPS; w++) {
            if (wm[w] == NEG_INF) continue; // a warp that saw no position
            float f = __expf(wm[w] - mx);
            v += wo[w][d] * f;
            tot += wl[w] * f;
        }
        out[(size_t)b * q_dim + h * hd + d] = v / tot;
    }
}

// attention_wide for WIDE_G query heads sharing a kv head at once: a block per (token, kv head, group of
// WIDE_G heads), each warp loading a position's key and value slice once for all of them. Every head's
// arithmetic is attention_wide's, step for step and in the same order, so the results are the same bits.
#define WIDE_G 4
__global__ void __launch_bounds__(256, 2) attention_wide_g(const float* __restrict__ qkv, const kv_t* __restrict__ kc, const kv_t* __restrict__ vc,
                                 const float* __restrict__ ks, const float* __restrict__ vs, float* __restrict__ out,
                                 int pos0, int n_heads, int n_kv, int hd, float scale) {
    __shared__ float wm[WIDE_WARPS], wl[WIDE_WARPS], wo[WIDE_WARPS][256];
    int b = blockIdx.x, kvh = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int G = n_heads / n_kv, h0 = kvh * G + blockIdx.z * WIDE_G;
    int q_dim = n_heads * hd, kv_dim = n_kv * hd, per = hd >> 5;
    float qr[WIDE_G][8], acc[WIDE_G][8], m[WIDE_G], l[WIDE_G];
    #pragma unroll
    for (int g = 0; g < WIDE_G; g++) {
        const float* q = qkv + (size_t)b * (q_dim + 2 * kv_dim) + (h0 + g) * hd + lane * per;
        #pragma unroll
        for (int i = 0; i < 8; i++) { qr[g][i] = i < per ? __fmul_rn(q[i], scale) : 0.f; acc[g][i] = 0.f; }
        m[g] = NEG_INF, l[g] = 0.f;
    }
    int span = pos0 + b + 1;
    for (int t = warp; t < span; t += WIDE_WARPS) {
        const kv_t* kt = kc + (size_t)t * kv_dim + kvh * hd + lane * per;
        const kv_t* vt = vc + (size_t)t * kv_dim + kvh * hd + lane * per;
        float4 k4[2], v4[2];
        #pragma unroll
        for (int j = 0; j < 2; j++) if (4 * j < per) { k4[j] = kv_load4(kt + 4 * j); v4[j] = kv_load4(vt + 4 * j); }
        float kscale = KV_SCALE(ks, t, n_kv, kvh), vscale = KV_SCALE(vs, t, n_kv, kvh);
        #pragma unroll
        for (int g = 0; g < WIDE_G; g++) {
            // attention_wide's expressions as NVRTC 12 contracts them (read from its SASS), spelled out so
            // that this kernel cannot contract them differently
            float s = 0.f;
            #pragma unroll
            for (int i = 0; i < 8; i += 4) {
                if (i < per) {
                    float4 k = k4[i >> 2];
                    s = __fadd_rn(s, __fmaf_rn(qr[g][i + 3], k.w, __fmaf_rn(qr[g][i + 2], k.z, __fmaf_rn(qr[g][i], k.x, __fmul_rn(qr[g][i + 1], k.y)))));
                }
            }
            for (int o = 16; o > 0; o >>= 1) s = __fadd_rn(s, __shfl_xor_sync(0xffffffff, s, o));
            s = __fmul_rn(s, kscale);
            float mn = fmaxf(m[g], s), c = __expf(__fsub_rn(m[g], mn)), p = __expf(__fsub_rn(s, mn));
            l[g] = __fmaf_rn(l[g], c, p);
            float pv = __fmul_rn(p, vscale);
            #pragma unroll
            for (int i = 0; i < 8; i += 4) {
                if (i < per) {
                    float4 v = v4[i >> 2];
                    acc[g][i] = __fmaf_rn(acc[g][i], c, __fmul_rn(pv, v.x)); acc[g][i + 1] = __fmaf_rn(acc[g][i + 1], c, __fmul_rn(pv, v.y));
                    acc[g][i + 2] = __fmaf_rn(pv, v.z, __fmul_rn(acc[g][i + 2], c)); acc[g][i + 3] = __fmaf_rn(pv, v.w, __fmul_rn(acc[g][i + 3], c));
                }
            }
            m[g] = mn;
        }
    }
    // merged a head at a time through one [warp][dim] buffer (small enough for two blocks an SM)
    #pragma unroll
    for (int g = 0; g < WIDE_G; g++) {
        if (g > 0) __syncthreads(); // the last head's merge is done reading wo
        if (lane == 0) { wm[warp] = m[g]; wl[warp] = l[g]; }
        #pragma unroll
        for (int i = 0; i < 8; i++) if (i < per) wo[warp][lane * per + i] = acc[g][i];
        __syncthreads();
        float mx = NEG_INF;
        for (int w = 0; w < WIDE_WARPS; w++) mx = fmaxf(mx, wm[w]);
        for (int d = threadIdx.x; d < hd; d += blockDim.x) {
            float v = 0.f, tot = 0.f;
            for (int w = 0; w < WIDE_WARPS; w++) {
                if (wm[w] == NEG_INF) continue; // a warp that saw no position
                float f = __expf(__fsub_rn(wm[w], mx));
                v = __fmaf_rn(wo[w][d], f, v);
                tot = __fmaf_rn(wl[w], f, tot);
            }
            out[(size_t)b * q_dim + (h0 + g) * hd + d] = __fdiv_rn(v, tot);
        }
    }
}

// x[i] *= sigmoid(g[i]) over n values: the attention output through its gates
__global__ void sigmoid_mul(float* __restrict__ x, const float* __restrict__ g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 1.f / (1.f + __expf(-g[i]));
}

// DeltaNet's causal convolution and SiLU, in place on the [q | k | v] part of the projected rows (stride floats
// apart), token after token; state: each channel's last `taps` inputs (oldest first), carried over. A thread per
// channel.
__global__ void delta_conv(float* __restrict__ x, int stride, const float* __restrict__ w, float* __restrict__ state,
                           int n, int channels, int taps) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float win[7];
    for (int i = 0; i < taps; i++) win[i] = state[(size_t)c * taps + i];
    const float* wc = w + (size_t)c * (taps + 1);
    for (int t = 0; t < n; t++) {
        float* p = x + (size_t)t * stride + c;
        float v = *p, acc = wc[taps] * v;
        for (int i = 0; i < taps; i++) acc += wc[i] * win[i];
        for (int i = 0; i + 1 < taps; i++) win[i] = win[i + 1];
        win[taps - 1] = v;
        *p = acc / (1.f + __expf(-acc));
    }
    for (int i = 0; i < taps; i++) state[(size_t)c * taps + i] = win[i];
}

// Per token: unit-length q and k heads (as ggml's l2_norm, x / max(|x|, eps); q also scaled), and each value
// head's beta = sigmoid(b) and decay = exp(softplus(a + dt_bias) * A) into gates (pairs), from the ba part of the
// row (per key head: its value heads' b, then their a; tiled: every head's b, then every head's a). grid
// (tokens, 2 * k_heads + 1), block = hd.
__global__ void delta_prep(float* __restrict__ x, int stride, int ba_off, const float* __restrict__ dt_bias, const float* __restrict__ A,
                           float* __restrict__ gates, int k_heads, int v_heads, int hd, float qscale, float eps, int tiled) {
    __shared__ float sh[32];
    int b = blockIdx.x, j = blockIdx.y;
    float* row = x + (size_t)b * stride;
    if (j < 2 * k_heads) {
        float* v = row + j * hd; // q heads, then k heads
        float ss = 0.f;
        for (int d = threadIdx.x; d < hd; d += blockDim.x) ss += v[d] * v[d];
        ss = block_sum(ss, sh);
        float f = (j < k_heads ? qscale : 1.f) / fmaxf(sqrtf(ss), eps);
        for (int d = threadIdx.x; d < hd; d += blockDim.x) v[d] *= f;
        return;
    }
    int per = v_heads / k_heads;
    const float* ba = row + ba_off;
    for (int h = threadIdx.x; h < v_heads; h += blockDim.x) {
        int bi = tiled ? h : h / per * 2 * per + h % per, ai = tiled ? v_heads + h : bi + per;
        float a = ba[ai] + dt_bias[h];
        float sp = a > 20.f ? a : log1pf(__expf(a));
        gates[((size_t)b * v_heads + h) * 2] = 1.f / (1.f + __expf(-ba[bi]));
        gates[((size_t)b * v_heads + h) * 2 + 1] = __expf(sp * A[h]);
    }
}

// The DeltaNet recurrence, token after token. Each thread owns one value column of a head's state (DN_HD
// floats, in registers): decay it, correct what it recalls for k towards v by beta, read it with q. state:
// [v head][key][value] (a warp's columns side by side, so its loads coalesce); out: n x (v_heads * DN_HD).
// Value head h reads key head h % k_heads if tiled, else h / (v_heads / k_heads). grid (v_heads, DN_HD / 32),
// block 32.
#define DN_HD 128
__global__ void delta_scan(const float* __restrict__ x, int stride, const float* __restrict__ gates, float* __restrict__ state,
                           float* __restrict__ out, int n, int k_heads, int v_heads, int tiled) {
    __shared__ float qs[DN_HD], kk[DN_HD];
    int h = blockIdx.x, col = blockIdx.y * 32 + threadIdx.x;
    int g = tiled ? h % k_heads : h / (v_heads / k_heads), kd = k_heads * DN_HD;
    float* st = state + (size_t)h * DN_HD * DN_HD + col;
    float s[DN_HD];
    #pragma unroll
    for (int i = 0; i < DN_HD; i++) s[i] = st[i * DN_HD];
    for (int t = 0; t < n; t++) {
        const float* row = x + (size_t)t * stride;
        __syncwarp();
        for (int i = threadIdx.x; i < DN_HD; i += 32) { qs[i] = row[g * DN_HD + i]; kk[i] = row[kd + g * DN_HD + i]; }
        __syncwarp();
        float beta = gates[((size_t)t * v_heads + h) * 2], decay = gates[((size_t)t * v_heads + h) * 2 + 1];
        float sk = 0.f, o = 0.f;
        #pragma unroll
        for (int i = 0; i < DN_HD; i++) { s[i] *= decay; sk += s[i] * kk[i]; }
        float dl = (row[2 * kd + h * DN_HD + col] - sk) * beta;
        #pragma unroll
        for (int i = 0; i < DN_HD; i++) { s[i] += kk[i] * dl; o += s[i] * qs[i]; }
        out[(size_t)t * v_heads * DN_HD + h * DN_HD + col] = o;
    }
    #pragma unroll
    for (int i = 0; i < DN_HD; i++) st[i * DN_HD] = s[i];
}

// Gated RMSNorm of each value head's output, in place: o / rms(o) * w * silu(z), z from the projected rows
// (x + z_off, stride floats apart). grid (tokens, v_heads), block = hd.
__global__ void delta_norm(float* __restrict__ o, const float* __restrict__ x, int stride, int z_off, const float* __restrict__ w,
                           int v_heads, int hd, float eps) {
    __shared__ float sh[32];
    int b = blockIdx.x, h = blockIdx.y;
    float* v = o + ((size_t)b * v_heads + h) * hd;
    const float* z = x + (size_t)b * stride + z_off + h * hd;
    float ss = 0.f;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) ss += v[d] * v[d];
    ss = block_sum(ss, sh);
    float inv = rsqrtf(ss / hd + eps);
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float zz = z[d];
        v[d] = v[d] * inv * w[d] * zz / (1.f + __expf(-zz));
    }
}

// x[b] += sigmoid(w · xb[b]) * y[b]: the shared expert's output through its gate. grid (tokens), block 256.
__global__ void gated_add(float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ w, const float* __restrict__ xb, int dim) {
    __shared__ float sh[32];
    int b = blockIdx.x;
    float s = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) s += w[i] * xb[(size_t)b * dim + i];
    s = block_sum(s, sh);
    float g = 1.f / (1.f + __expf(-s));
    for (int i = threadIdx.x; i < dim; i += blockDim.x) x[(size_t)b * dim + i] += g * y[(size_t)b * dim + i];
}

// ---- 2-bit weights (IQ2_XXS: fmt 0, 66-byte blocks; IQ2_S: fmt 1, 82-byte blocks; IQ2_XS: fmt 2, 74-byte blocks;
// see iquant.rs). Blocks are only 2-byte aligned, so they are read a byte at a time. A lane takes one group of 8
// values of a 256-value block: 4 groups per 32 (lanes 4*ib .. 4*ib+3 cover 32-value sub-block ib).

__device__ __forceinline__ int iq2_bytes(int fmt) { return fmt == 0 ? 66 : fmt == 1 ? 82 : 74; }

// byte masks: 0xff in byte i where bit i of the 4-bit argument is set
__device__ __forceinline__ unsigned iq2_mask4(unsigned bits) {
    return ((bits & 1) ? 0xFFu : 0u) | ((bits & 2) ? 0xFF00u : 0u) | ((bits & 4) ? 0xFF0000u : 0u) | ((bits & 8) ? 0xFF000000u : 0u);
}

// One group of 8 values of block `blk` (lane's group g = 4 * ib + l): its grid entry, sign byte and scale.
__device__ __forceinline__ void iq2_group(int fmt, const unsigned char* blk, int g, unsigned long long* grid, unsigned* signs, float* scale) {
    float d = h2f((unsigned)blk[0] | ((unsigned)blk[1] << 8));
    int ib = g >> 2, l = g & 3;
    if (fmt == 0) {
        const unsigned char* q = blk + 2 + 8 * ib;
        unsigned aux = (unsigned)q[4] | ((unsigned)q[5] << 8) | ((unsigned)q[6] << 16) | ((unsigned)q[7] << 24);
        *grid = iq2xxs_grid_d[q[l]];
        *signs = ksigns_iq2xs_d[(aux >> (7 * l)) & 127];
        *scale = d * (0.5f + (float)(aux >> 28)) * 0.25f;
    } else if (fmt == 2) {
        unsigned q = (unsigned)blk[2 + 2 * g] | ((unsigned)blk[3 + 2 * g] << 8);
        *grid = iq2xs_grid_d[q & 511];
        *signs = ksigns_iq2xs_d[q >> 9];
        *scale = d * (0.5f + (float)((blk[66 + ib] >> (4 * (l >> 1))) & 0xF)) * 0.25f;
    } else {
        int idx = blk[2 + 4 * ib + l] | ((blk[66 + ib] << (8 - 2 * l)) & 0x300);
        *grid = iq2s_grid_d[idx];
        *signs = blk[34 + 4 * ib + l];
        *scale = d * (0.5f + (float)((blk[74 + ib] >> (4 * (l >> 1))) & 0xF)) * 0.25f;
    }
}

// Lane's share of one row (nblk blocks of 256) dotted with int8 activations (32-value blocks: xq values, xd scales).
__device__ __forceinline__ float iq2_lane_dot(int fmt, const unsigned char* row, int nblk, const signed char* xq, const float* xd, int lane) {
    int bytes = iq2_bytes(fmt);
    float acc = 0.f;
    for (int b = 0; b < nblk; b++) {
        unsigned long long grid;
        unsigned signs;
        float scale;
        iq2_group(fmt, row + (size_t)b * bytes, lane, &grid, &signs, &scale);
        unsigned m0 = iq2_mask4(signs & 15), m1 = iq2_mask4(signs >> 4);
        // grid levels (<= 43) signed by the mask: (g ^ 0xff) - 0xff = -g per byte
        int w0 = (int)__vsub4((unsigned)grid ^ m0, m0), w1 = (int)__vsub4((unsigned)(grid >> 32) ^ m1, m1);
        int xb = b * 8 + (lane >> 2);
        const int* xv = reinterpret_cast<const int*>(xq + (size_t)xb * 32 + 8 * (lane & 3));
        acc += scale * xd[xb] * (float)__dp4a(w1, xv[1], __dp4a(w0, xv[0], 0));
    }
    return acc;
}

// Mixture of experts, one token, the 2-bit experts held in GPU memory (the table as moe_gate_up_q3k's):
// gate / up rows, a warp per row.
__global__ void moe_gate_up_iq2(const unsigned long long* __restrict__ ptrs, const signed char* __restrict__ xq, const float* __restrict__ xd,
                                float* __restrict__ out, int rows, int cols, int fmt) {
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= rows) return;
    int nblk = cols >> 8;
    const unsigned char* w = reinterpret_cast<const unsigned char*>(ptrs[blockIdx.y]) + (size_t)row * nblk * iq2_bytes(fmt);
    float v = warp_sum(iq2_lane_dot(fmt, w, nblk, xq, xd, lane));
    if (lane == 0) out[(size_t)blockIdx.y * rows + row] = v;
}

// down: expert k reads its own activations (row k of xq / xd, written by silu_q8)
__global__ void moe_down_iq2(const unsigned long long* __restrict__ ptrs, int nk, const signed char* __restrict__ xq, const float* __restrict__ xd,
                             float* __restrict__ out, int rows, int cols, int fmt) {
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31, k = blockIdx.y;
    if (row >= rows) return;
    int nblk = cols >> 8;
    const unsigned char* w = reinterpret_cast<const unsigned char*>(ptrs[2 * nk + k]) + (size_t)row * nblk * iq2_bytes(fmt);
    float v = warp_sum(iq2_lane_dot(fmt, w, nblk, xq + (size_t)k * cols, xd + (size_t)k * (cols >> 5), lane));
    if (lane == 0) out[(size_t)k * rows + row] = v;
}

// A dense 2-bit matrix (decoding): out[t * out_stride + out_off + row] (+)= W[row] · x[t] over n tokens of int8
// activations. A warp per row; grid (rows / 8, n).
__global__ void matmul_iq2_dp4a(const unsigned char* __restrict__ w, const signed char* __restrict__ xq, const float* __restrict__ xd,
                                float* __restrict__ out, int rows, int cols, int accumulate, int out_stride, int out_off, int fmt) {
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31, t = blockIdx.y;
    if (row >= rows) return;
    int nblk = cols >> 8;
    float v = warp_sum(iq2_lane_dot(fmt, w + (size_t)row * nblk * iq2_bytes(fmt), nblk, xq + (size_t)t * cols, xd + (size_t)t * (cols >> 5), lane));
    if (lane == 0) {
        size_t i = (size_t)t * out_stride + out_off + row;
        out[i] = accumulate ? out[i] + v : v;
    }
}

// 2-bit matrix to Q8 (int8 per value and an f32 scale per 32, requantised) for the prompt-batch matmuls.
// One warp per 32 values: lane j of group j / 8.
__global__ void iq2_to_q8(const unsigned char* __restrict__ w, signed char* __restrict__ q, float* __restrict__ d, int blocks32, int fmt) {
    int b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (b >= blocks32) return;
    unsigned long long grid;
    unsigned signs;
    float scale;
    iq2_group(fmt, w + (size_t)(b >> 3) * iq2_bytes(fmt), 4 * (b & 7) + (lane >> 3), &grid, &signs, &scale);
    int j = lane & 7;
    float v = scale * (float)((grid >> (8 * j)) & 0xFF) * ((signs >> j) & 1 ? -1.f : 1.f);
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    float dd = amax / 127.f;
    q[(size_t)b * 32 + lane] = (signed char)__float2int_rn(dd != 0.f ? v / dd : 0.f);
    if (lane == 0) d[b] = dd;
}

// ---- CSM speech (see CsmFrames) ----

__device__ __forceinline__ unsigned long long splitmix(unsigned long long z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// One code from n <= 4096 logits: temperature and top-k, drawn with the Gumbel-max trick (the largest
// logit / temperature + Gumbel noise among the k largest is a draw from their softmax). The k-th largest
// value is found by a radix select over order-preserving keys; ties with it stay in, as top-k filters do.
// p: [k, 1 / temperature (0: greedy)]; rng: [seed, draws so far], moved on. One block of 1024 threads.
__global__ void csm_sample(const float* __restrict__ logits, int n, const float* __restrict__ p, unsigned long long* rng,
                           int* __restrict__ codes, int slot) {
    __shared__ unsigned int keys[4096];
    __shared__ int count;
    __shared__ float sv[32];
    __shared__ int si[32];
    int k = (int)p[0];
    float inv_t = p[1];
    unsigned long long seed = rng[0], draw = rng[1];
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        unsigned int u = __float_as_uint(logits[i]);
        keys[i] = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    }
    unsigned int th = 0;
    if (inv_t > 0.f && k > 0 && k < n) {
        for (int b = 31; b >= 0; b--) {
            unsigned int cand = th | (1u << b);
            if (threadIdx.x == 0) count = 0;
            __syncthreads();
            int c = 0;
            for (int i = threadIdx.x; i < n; i += blockDim.x) c += keys[i] >= cand;
            for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xffffffff, c, o);
            if ((threadIdx.x & 31) == 0 && c) atomicAdd(&count, c);
            __syncthreads();
            if (count >= k) th = cand;
            __syncthreads(); // everyone has read count before it is reset
        }
    }
    __syncthreads();
    float bv = NEG_INF;
    int bi = 0x7FFFFFFF;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (keys[i] < th) continue;
        float v = logits[i];
        if (inv_t > 0.f) {
            unsigned long long h = splitmix(seed ^ splitmix(draw * 4096ull + (unsigned long long)i));
            float u = ((float)(h >> 40) + 0.5f) * (1.f / 16777216.f);
            v = v * inv_t - logf(-logf(u));
        }
        if (v > bv) { bv = v; bi = i; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, bv, o);
        int oi = __shfl_xor_sync(0xffffffff, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { sv[wid] = bv; si[wid] = bi; }
    __syncthreads();
    if (wid == 0) {
        int nw = blockDim.x >> 5;
        bv = lane < nw ? sv[lane] : NEG_INF;
        bi = lane < nw ? si[lane] : 0x7FFFFFFF;
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, bv, o);
            int oi = __shfl_xor_sync(0xffffffff, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) {
            codes[slot] = bi == 0x7FFFFFFF ? 0 : bi;
            rng[1] = draw + 1;
        }
    }
}

// out = the sum over codebooks first .. first + m - 1 of the f16 table row (codebook * vocab + its code).
__global__ void csm_embed(const unsigned short* __restrict__ table, int cols, const int* __restrict__ codes, int first, int m,
                          int vocab, float* __restrict__ out) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    float acc = 0.f;
    for (int j = first; j < first + m; j++) acc += h2f(table[((size_t)j * vocab + codes[j]) * cols + c]);
    out[c] = acc;
}

// ---- Whisper (see GpuWhisper) ----

// LayerNorm with a bias over rows of `dim`: one block per row.
__global__ void wh_layernorm(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, float* __restrict__ out, int dim, float eps) {
    __shared__ float sh[32];
    const float* r = x + (size_t)blockIdx.x * dim;
    float* o = out + (size_t)blockIdx.x * dim;
    float s = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) s += r[i];
    float mean = block_sum(s, sh) / dim;
    float v = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) { float d = r[i] - mean; v += d * d; }
    float inv = rsqrtf(block_sum(v, sh) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) o[i] = (r[i] - mean) * inv * w[i] + b[i];
}

// x[i] += b[i % dim] over n rows, then the exact GELU if `gelu`.
__global__ void wh_bias(float* __restrict__ x, const float* __restrict__ b, int dim, int n, int gelu) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * dim) return;
    float v = x[i] + b[i % dim];
    if (gelu) v = 0.5f * v * (1.f + erff(v * 0.70710678f));
    x[i] = v;
}

__global__ void wh_add(float* __restrict__ x, const float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// Rows for a 3-tap convolution with one zero of padding each side: output row t is [x(t*s-1) | x(t*s) | x(t*s+1)]
// over `c` channels each (zeros outside the nin input rows).
__global__ void wh_im2col(const float* __restrict__ x, float* __restrict__ out, int nin, int nout, int c, int stride) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)nout * 3 * c) return;
    int t = (int)(i / (3 * c)), r = (int)(i % (3 * c)), j = r / c, ch = r % c;
    int src = t * stride - 1 + j;
    out[i] = (src >= 0 && src < nin) ? x[(size_t)src * c + ch] : 0.f;
}

// Attention with heads of 64: query row qi (stride qs) of head h against nk key and value rows (stride ks);
// causal >= 0: query qi sees keys below causal + qi + 1. grid (queries, heads), 128 threads, nk floats of
// dynamic shared memory.
__global__ void wh_attention(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, float* __restrict__ out,
                             int nk, int qs, int ks, int heads, int causal, float scale) {
    extern __shared__ float sc[];
    __shared__ float qv[64];
    __shared__ float sh[32];
    int qi = blockIdx.x, h = blockIdx.y;
    if (threadIdx.x < 64) qv[threadIdx.x] = q[(size_t)qi * qs + h * 64 + threadIdx.x];
    __syncthreads();
    int lim = causal >= 0 ? min(nk, causal + qi + 1) : nk;
    float mx = NEG_INF;
    for (int j = threadIdx.x; j < lim; j += blockDim.x) {
        const float* kr = k + (size_t)j * ks + h * 64;
        float s = 0.f;
        for (int d = 0; d < 64; d++) s += qv[d] * kr[d];
        s *= scale;
        sc[j] = s;
        mx = fmaxf(mx, s);
    }
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int j = threadIdx.x; j < lim; j += blockDim.x) { float e = __expf(sc[j] - mx); sc[j] = e; sum += e; }
    sum = block_sum(sum, sh);
    if (threadIdx.x < 64) {
        float acc = 0.f;
        for (int j = 0; j < lim; j++) acc += sc[j] * v[(size_t)j * ks + h * 64 + threadIdx.x];
        out[(size_t)qi * heads * 64 + h * 64 + threadIdx.x] = acc / sum;
    }
}

}
"#);

// ---------------------------------------------------------------- dynamic loading

type CuResult = i32;
type Ptr = *mut c_void;

#[cfg(windows)]
mod os {
    use std::ffi::c_void;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
        fn SetDllDirectoryW(dir: *const u16) -> i32;
        fn GetCurrentProcess() -> *mut c_void;
        fn PrefetchVirtualMemory(process: *mut c_void, n: usize, ranges: *const Range, flags: u32) -> i32;
        fn VirtualUnlock(addr: *mut c_void, bytes: usize) -> i32;
    }
    const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x8;

    #[repr(C)]
    struct Range {
        addr: *mut c_void,
        bytes: usize,
    }

    /// Asks Windows to read these (address, length) ranges of mapped memory in now, as large concurrent
    /// reads, instead of one 4 KB page fault at a time later. Only a hint: failure changes nothing.
    pub fn prefetch(ranges: &[(usize, usize)]) {
        let r: Vec<Range> = ranges.iter().map(|&(a, n)| Range { addr: a as *mut c_void, bytes: n }).collect();
        unsafe { PrefetchVirtualMemory(GetCurrentProcess(), r.len(), r.as_ptr(), 0) };
    }

    /// This process's page faults so far and its working set in bytes.
    pub fn page_faults() -> (u64, u64) {
        #[repr(C)]
        #[derive(Default)]
        struct Pmc {
            cb: u32,
            faults: u32,
            peak_ws: usize,
            ws: usize,
            rest: [usize; 6],
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn K32GetProcessMemoryInfo(p: *mut c_void, c: *mut Pmc, cb: u32) -> i32;
        }
        let mut c = Pmc { cb: size_of::<Pmc>() as u32, ..Default::default() };
        unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
        (c.faults as u64, c.ws as u64)
    }

    /// Takes these ranges of mapped memory out of the working set (VirtualUnlock on pages that are not
    /// locked does exactly that): they stay in the file cache, but are the first memory Windows reuses.
    pub fn release(ranges: &[(usize, usize)]) {
        for &(a, n) in ranges {
            unsafe { VirtualUnlock(a as *mut c_void, n) };
        }
    }

    pub fn open(path: &str) -> Option<*mut c_void> {
        // nvrtc loads nvrtc-builtins*.dll lazily at compile time; make its folder searchable
        if let Some(dir) = std::path::Path::new(path).parent() {
            let wide: Vec<u16> = dir.to_string_lossy().encode_utf16().chain(Some(0)).collect();
            unsafe { SetDllDirectoryW(wide.as_ptr()) };
        }
        let wide: Vec<u16> = path.encode_utf16().chain(Some(0)).collect();
        let h = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH) };
        (!h.is_null()).then_some(h)
    }

    pub fn symbol(lib: *mut c_void, name: &str) -> Option<*mut c_void> {
        let c = std::ffi::CString::new(name).ok()?;
        let p = unsafe { GetProcAddress(lib, c.as_ptr() as *const u8) };
        (!p.is_null()).then_some(p)
    }
}

#[cfg(not(windows))]
mod os {
    use std::ffi::c_void;
    pub fn open(_: &str) -> Option<*mut c_void> {
        None
    }
    pub fn symbol(_: *mut c_void, _: &str) -> Option<*mut c_void> {
        None
    }
    pub fn prefetch(_: &[(usize, usize)]) {}
    pub fn release(_: &[(usize, usize)]) {}
    pub fn page_faults() -> (u64, u64) {
        (0, 0)
    }
}

macro_rules! api {
    ($($name:ident : fn($($arg:ty),*) -> $ret:ty;)*) => {
        #[allow(non_snake_case)]
        struct Fns { $($name: unsafe extern "system" fn($($arg),*) -> $ret,)* }
        impl Fns {
            fn load(cuda: Ptr, nvrtc: Ptr) -> Result<Fns, String> {
                Ok(Fns { $($name: {
                    let lib = if stringify!($name).starts_with("nvrtc") { nvrtc } else { cuda };
                    let p = os::symbol(lib, stringify!($name)).ok_or_else(|| format!("missing CUDA symbol {}", stringify!($name)))?;
                    unsafe { std::mem::transmute::<*mut c_void, unsafe extern "system" fn($($arg),*) -> $ret>(p) }
                },)* })
            }
        }
    };
}

api! {
    cuInit: fn(u32) -> CuResult;
    cuDeviceGet: fn(*mut i32, i32) -> CuResult;
    cuDeviceGetAttribute: fn(*mut i32, i32, i32) -> CuResult;
    cuDeviceGetName: fn(*mut c_char, i32, i32) -> CuResult;
    cuCtxCreate_v2: fn(*mut Ptr, u32, i32) -> CuResult;
    cuCtxSetCurrent: fn(Ptr) -> CuResult;
    cuModuleLoadData: fn(*mut Ptr, *const c_void) -> CuResult;
    cuModuleGetFunction: fn(*mut Ptr, Ptr, *const c_char) -> CuResult;
    cuMemAlloc_v2: fn(*mut u64, usize) -> CuResult;
    cuMemFree_v2: fn(u64) -> CuResult;
    cuMemGetInfo_v2: fn(*mut usize, *mut usize) -> CuResult;
    cuMemGetAllocationGranularity: fn(*mut usize, *const MemProp, i32) -> CuResult;
    cuMemAddressReserve: fn(*mut u64, usize, usize, u64, u64) -> CuResult;
    cuMemAddressFree: fn(u64, usize) -> CuResult;
    cuMemCreate: fn(*mut u64, usize, *const MemProp, u64) -> CuResult;
    cuMemRelease: fn(u64) -> CuResult;
    cuMemMap: fn(u64, usize, usize, u64, u64) -> CuResult;
    cuMemUnmap: fn(u64, usize) -> CuResult;
    cuMemSetAccess: fn(u64, usize, *const MemAccess, usize) -> CuResult;
    cuMemcpyHtoD_v2: fn(u64, *const c_void, usize) -> CuResult;
    cuMemcpyDtoH_v2: fn(*mut c_void, u64, usize) -> CuResult;
    cuMemcpyDtoD_v2: fn(u64, u64, usize) -> CuResult;
    cuLaunchKernel: fn(Ptr, u32, u32, u32, u32, u32, u32, u32, Ptr, *mut Ptr, *mut Ptr) -> CuResult;
    cuFuncSetAttribute: fn(Ptr, i32, i32) -> CuResult;
    cuFuncGetAttribute: fn(*mut i32, i32, Ptr) -> CuResult;
    cuCtxSynchronize: fn() -> CuResult;
    cuStreamCreate: fn(*mut Ptr, u32) -> CuResult;
    cuStreamSynchronize: fn(Ptr) -> CuResult;
    cuStreamDestroy_v2: fn(Ptr) -> CuResult;
    cuMemAllocHost_v2: fn(*mut *mut c_void, usize) -> CuResult;
    cuMemFreeHost: fn(*mut c_void) -> CuResult;
    cuMemcpyHtoDAsync_v2: fn(u64, *const c_void, usize, Ptr) -> CuResult;
    cuMemcpyDtoHAsync_v2: fn(*mut c_void, u64, usize, Ptr) -> CuResult;
    cuEventCreate: fn(*mut Ptr, u32) -> CuResult;
    cuEventRecord: fn(Ptr, Ptr) -> CuResult;
    cuEventSynchronize: fn(Ptr) -> CuResult;
    cuEventDestroy_v2: fn(Ptr) -> CuResult;
    cuStreamBeginCapture_v2: fn(Ptr, i32) -> CuResult;
    cuStreamEndCapture: fn(Ptr, *mut Ptr) -> CuResult;
    cuGraphInstantiateWithFlags: fn(*mut Ptr, Ptr, u64) -> CuResult;
    cuGraphLaunch: fn(Ptr, Ptr) -> CuResult;
    cuGraphDestroy: fn(Ptr) -> CuResult;
    cuGraphExecDestroy: fn(Ptr) -> CuResult;
    cuGetErrorString: fn(CuResult, *mut *const c_char) -> CuResult;
    nvrtcCreateProgram: fn(*mut Ptr, *const c_char, *const c_char, i32, *const *const c_char, *const *const c_char) -> i32;
    nvrtcCompileProgram: fn(Ptr, i32, *const *const c_char) -> i32;
    nvrtcGetCUBINSize: fn(Ptr, *mut usize) -> i32;
    nvrtcGetCUBIN: fn(Ptr, *mut c_char) -> i32;
    nvrtcGetProgramLogSize: fn(Ptr, *mut usize) -> i32;
    nvrtcGetProgramLog: fn(Ptr, *mut c_char) -> i32;
    nvrtcDestroyProgram: fn(*mut Ptr) -> i32;
}

struct Api {
    f: Fns,
    ctx: usize, // the context, for threads other than the one that made it
    pub device_name: String,
    pub compute: (i32, i32),
    pub nvrtc_path: String,
}

// the raw function pointers and library handles are process-global and immutable once loaded
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

impl Api {
    fn check(&self, r: CuResult, what: &str) -> Result<(), String> {
        if r == 0 {
            return Ok(());
        }
        let mut msg: *const c_char = std::ptr::null();
        unsafe { (self.f.cuGetErrorString)(r, &mut msg) };
        let text = if msg.is_null() { "unknown error".to_string() } else { unsafe { CStr::from_ptr(msg) }.to_string_lossy().into_owned() };
        Err(format!("CUDA {what} failed: {text} ({r})"))
    }
}

/// GHOST_INFER_NVRTC, else the NVRTC beside the program (the installer ships one), else the newest under the
/// CUDA toolkit folder.
fn find_nvrtc() -> Option<String> {
    if let Ok(p) = std::env::var("GHOST_INFER_NVRTC") {
        return Some(p);
    }
    let is_nvrtc = |p: &std::path::Path| {
        let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        n.starts_with("nvrtc64_") && n.ends_with(".dll") && !n.contains(".alt")
    };
    let beside = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.to_path_buf()));
    if let Some(p) = beside.and_then(|d| std::fs::read_dir(d).ok()).and_then(|d| d.flatten().map(|e| e.path()).filter(|p| is_nvrtc(p)).max()) {
        return Some(p.to_string_lossy().into_owned());
    }
    let mut roots = Vec::new();
    if let Ok(p) = std::env::var("CUDA_PATH") {
        roots.push(std::path::PathBuf::from(&p));
        roots.push(std::path::PathBuf::from(&p).join("bin"));
    }
    if let Ok(dirs) = std::fs::read_dir(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA") {
        roots.extend(dirs.flatten().map(|d| d.path().join("bin")));
    }
    let mut found: Vec<std::path::PathBuf> = roots
        .iter()
        .filter_map(|r| std::fs::read_dir(r).ok())
        .flat_map(|d| d.flatten().map(|e| e.path()))
        .filter(|p| is_nvrtc(p))
        .collect();
    found.sort();
    found.pop().map(|p| p.to_string_lossy().into_owned())
}

fn api() -> Result<&'static Api, String> {
    static API: OnceLock<Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let cuda = os::open("nvcuda.dll").ok_or("no NVIDIA driver (nvcuda.dll) found")?;
        let nvrtc_path = find_nvrtc().ok_or("no NVRTC found (install the CUDA toolkit or set GHOST_INFER_NVRTC)")?;
        let nvrtc = os::open(&nvrtc_path).ok_or_else(|| format!("cannot load {nvrtc_path}"))?;
        let f = Fns::load(cuda, nvrtc)?;
        let mut api = Api { f, ctx: 0, device_name: String::new(), compute: (0, 0), nvrtc_path };
        unsafe {
            api.check((api.f.cuInit)(0), "init")?;
            let mut dev = 0;
            api.check((api.f.cuDeviceGet)(&mut dev, 0), "device")?;
            let mut name = [0 as c_char; 256];
            api.check((api.f.cuDeviceGetName)(name.as_mut_ptr(), 256, dev), "device name")?;
            api.device_name = CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();
            let (mut major, mut minor) = (0, 0);
            api.check((api.f.cuDeviceGetAttribute)(&mut major, 75, dev), "compute capability")?;
            api.check((api.f.cuDeviceGetAttribute)(&mut minor, 76, dev), "compute capability")?;
            api.compute = (major, minor);
            let mut ctx: Ptr = std::ptr::null_mut();
            api.check((api.f.cuCtxCreate_v2)(&mut ctx, 0, dev), "context")?;
            api.ctx = ctx as usize;
        }
        Ok(api)
    })
    .as_ref()
    .map_err(|e| e.clone())
}

// ---------------------------------------------------------------- device memory and kernels

struct DeviceBuf {
    ptr: u64,
    bytes: usize,
    owned: bool, // false: a part of an Arena's range, freed with it
}

impl DeviceBuf {
    fn alloc(bytes: usize) -> Result<DeviceBuf, String> {
        let a = api()?;
        let mut ptr = 0u64;
        a.check(unsafe { (a.f.cuMemAlloc_v2)(&mut ptr, bytes.max(4)) }, &format!("alloc {} MB", bytes >> 20))?;
        Ok(DeviceBuf { ptr, bytes, owned: true })
    }

    fn write<T: Copy>(&self, data: &[T]) -> Result<(), String> {
        let a = api()?;
        let bytes = std::mem::size_of_val(data);
        assert!(bytes <= self.bytes);
        a.check(unsafe { (a.f.cuMemcpyHtoD_v2)(self.ptr, data.as_ptr() as *const c_void, bytes) }, "copy to GPU")
    }

    /// Byte offset into the buffer, for passing a sub-range to a kernel.
    fn at(&self, bytes: usize) -> u64 {
        self.ptr + bytes as u64
    }
}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        if !self.owned {
            return;
        }
        if let Ok(a) = api() {
            unsafe { (a.f.cuMemFree_v2)(self.ptr) };
        }
    }
}

/// CUmemAllocationProp: pinned device memory on the device.
#[repr(C)]
struct MemProp {
    ty: i32,
    handle_types: i32,
    loc_type: i32,
    loc_id: i32,
    win32_meta: *mut c_void,
    compression: u8,
    rdma: u8,
    usage: u16,
    reserved: [u8; 4],
}

/// CUmemAccessDesc
#[repr(C)]
struct MemAccess {
    loc_type: i32,
    loc_id: i32,
    flags: i32,
}

const ON_DEVICE: MemProp = MemProp { ty: 1, handle_types: 0, loc_type: 1, loc_id: 0, win32_meta: std::ptr::null_mut(), compression: 0, rdma: 0, usage: 0, reserved: [0; 4] };

/// A model's weights one after another in one range of addresses, given memory as it fills. cuMemAlloc rounds
/// every buffer up to 2 MB: over a 12B model's ~300 matrices and scale arrays that was 250 MB, a quarter of
/// what an 8 GB card has left for the context.
struct Arena {
    base: u64,
    reserved: usize,
    used: usize,
    mapped: usize,
    gran: usize,
    chunks: Vec<(u64, usize)>, // physical memory handles and their sizes, mapped in order from base
}

impl Arena {
    fn new() -> Result<Arena, String> {
        let a = api()?;
        let (mut gran, mut free, mut total, mut base) = (0usize, 0usize, 0usize, 0u64);
        unsafe {
            a.check((a.f.cuMemGetAllocationGranularity)(&mut gran, &ON_DEVICE, 0), "allocation granularity")?;
            a.check((a.f.cuMemGetInfo_v2)(&mut free, &mut total), "memory info")?;
            let reserved = total.next_multiple_of(gran);
            a.check((a.f.cuMemAddressReserve)(&mut base, reserved, 0, 0, 0), "address reserve")?;
            Ok(Arena { base, reserved, used: 0, mapped: 0, gran, chunks: Vec::new() })
        }
    }

    fn alloc(&mut self, bytes: usize) -> Result<DeviceBuf, String> {
        let start = self.used.next_multiple_of(256);
        let end = start + bytes.max(4);
        if end > self.reserved {
            return DeviceBuf::alloc(bytes);
        }
        if end > self.mapped {
            let a = api()?;
            let size = (end - self.mapped).next_multiple_of(self.gran);
            let at = self.base + self.mapped as u64;
            let mut h = 0u64;
            unsafe {
                a.check((a.f.cuMemCreate)(&mut h, size, &ON_DEVICE, 0), &format!("alloc {} MB", size >> 20))?;
                if let Err(e) = a.check((a.f.cuMemMap)(at, size, 0, h, 0), "map") {
                    (a.f.cuMemRelease)(h);
                    return Err(e);
                }
                self.chunks.push((h, size));
                self.mapped += size;
                let rw = MemAccess { loc_type: 1, loc_id: 0, flags: 3 };
                a.check((a.f.cuMemSetAccess)(at, size, &rw, 1), "memory access")?;
            }
        }
        self.used = end;
        Ok(DeviceBuf { ptr: self.base + start as u64, bytes, owned: false })
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        if let Ok(a) = api() {
            let mut at = self.base;
            unsafe {
                for &(h, size) in &self.chunks {
                    (a.f.cuMemUnmap)(at, size);
                    (a.f.cuMemRelease)(h);
                    at += size as u64;
                }
                (a.f.cuMemAddressFree)(self.base, self.reserved);
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Arg {
    P(u64),
    I(i32),
    F(f32),
}

struct Kernels {
    embed: Ptr,
    rmsnorm: Ptr,
    rmsnorm_q8: Ptr, // one token: RMSNorm straight to int8 activations
    qk_norm_rope: Ptr,
    matmul_q8: Ptr,
    matmul_q4k: [Ptr; 2], // f32 activations: one-token and four-token variants
    matmul_q6k: [Ptr; 2],
    matmul_dp4a: [[Ptr; 2]; 3], // decode, int8 activations: [Q8_0 | Q4_K | Q6_K][1 token | up to 8]
    argmax_rows: Ptr,
    csm_sample: Ptr,
    csm_embed: Ptr,
    wh_layernorm: Ptr,
    wh_bias: Ptr,
    wh_add: Ptr,
    wh_im2col: Ptr,
    wh_attention: Ptr,
    matmul_tiled: Ptr,
    matmul_mma: Option<Ptr>, // int8 tensor cores, sm_75 and newer
    matmul_mma_k: Option<[Ptr; 2]>, // the same for Q4_K / Q6_K with per-256 activation scales
    matmul_vfy: Option<[[Ptr; 3]; 2]>, // verify batches, exact per token: [Q4_K | Q6_K][8 | 16 | 32 tokens]
    quantize_q8: Ptr,
    quantize_q8k: Ptr,
    attention_partial: Ptr,
    attention_partial_gqa: Ptr,
    attention_vfy: Option<[[Ptr; 2]; 3]>, // verify batches, grouped-query, int8 cache: [2 | 4 | 8 tokens per block][scores, outputs]
    attention_combine: Ptr,
    attention_prefill: Ptr,
    attention_prefill_tc: Option<[Ptr; 2]>, // f16 tensor cores (sm_75+), head size 64 / 128
    attention_prefill_tc256: Option<Ptr>,   // the same for head size 256 (prompt batches of attention_wide models)
    attention_decode_tc256: Option<Ptr>,    // one-token steps and verify batches of 256-wide grouped heads, int8 cache
    silu_mul: Ptr,
    silu_q8: Ptr,
    add_rows: Ptr,
    moe_gate_up_q3k: Ptr,
    moe_down_q4k: Ptr,
    moe_combine: Ptr,
    router_logits: Ptr,
    router_logits1: Ptr,
    moe_gather: Ptr,
    moe_scatter: Ptr,
    q3k_to_q8: Ptr,
    q5k_to_q8: Ptr,
    // multi-head latent attention
    rmsnorm_rows: Ptr,
    mla_kv: Ptr,
    mla_q_rope: Ptr,
    head_matvec: Ptr,
    mla_attention: Ptr,
    mla_combine: Ptr,
    // Qwen3-Next
    next_split_qgate: Ptr,
    attention_wide: Ptr,
    attention_wide_g: Ptr,
    sigmoid_mul: Ptr,
    delta_conv: Ptr,
    delta_prep: Ptr,
    delta_scan: Ptr,
    delta_norm: Ptr,
    gated_add: Ptr,
    // 2-bit experts
    moe_gate_up_iq2: Ptr,
    moe_down_iq2: Ptr,
    iq2_to_q8: Ptr,
    matmul_iq2_dp4a: Ptr,
    matmul_q3k_dp4a: [Ptr; 2], // decode, Fmt::Q3K: [1 token | up to 8]
    /// every launch goes here (a blocking stream: the synchronous copies still order against it); CUDA
    /// graphs need a stream other than the legacy default one
    stream: Ptr,
    /// a non-blocking stream that copies go through too (`h2d`, `d2h`), never the legacy one: for models that run
    /// on threads of their own beside the language model (the voice, Whisper). Otherwise a synchronous copy on one
    /// thread synchronises with the other thread's blocking stream, and spoils a CUDA graph it is capturing.
    private: bool,
}

unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

impl Kernels {
    /// The kernels as a GPU binary (cubin) for the installed GPU: from the on-disk cache when the source,
    /// options, NVRTC and GPU match a previous run, else compiled with NVRTC (seconds) and cached.
    /// GHOST_INFER_NO_KERNEL_CACHE=1 always compiles.
    fn binary(a: &Api) -> Result<Vec<u8>, String> {
        let mut opts = vec![
            format!("--gpu-architecture=sm_{}{}", a.compute.0, a.compute.1),
            "-std=c++14".into(),
            format!("-DRPW={}", kq_rpw()),
            format!("-DATT_CHUNK_CAP={}", att_chunk_cap()),
            format!("-DQCH={}", qch()),
            format!("-DDP4A_BATCH_MINB={}", std::env::var("GHOST_INFER_BATCH_MINB").unwrap_or("3".into())),
            format!("-DDP4A_BATCH_RPW={}", batch_rpw()),
        ];
        if kv_int8() {
            opts.push("-DKV8".into());
        }
        let key = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (KERNELS, &opts, &a.nvrtc_path, &a.device_name).hash(&mut h);
            format!("{:016x}", h.finish())
        };
        let use_cache = std::env::var_os("GHOST_INFER_NO_KERNEL_CACHE").is_none();
        let dir = std::env::var_os("LOCALAPPDATA").map(|d| std::path::PathBuf::from(d).join("ghost-infer").join("kernels"));
        let file = dir.as_ref().map(|d| d.join(format!("{key}.cubin")));
        if let (true, Some(file)) = (use_cache, &file) {
            if let Ok(bin) = std::fs::read(file) {
                return Ok(bin);
            }
        }
        let f = &a.f;
        let bin = unsafe {
            let src = CString::new(KERNELS).unwrap();
            let name = CString::new("ghost-infer.cu").unwrap();
            let mut prog: Ptr = std::ptr::null_mut();
            if (f.nvrtcCreateProgram)(&mut prog, src.as_ptr(), name.as_ptr(), 0, std::ptr::null(), std::ptr::null()) != 0 {
                return Err("nvrtcCreateProgram failed".into());
            }
            let copts: Vec<CString> = opts.iter().map(|o| CString::new(o.as_str()).unwrap()).collect();
            let opt_ptrs: Vec<*const c_char> = copts.iter().map(|o| o.as_ptr()).collect();
            let rc = (f.nvrtcCompileProgram)(prog, opt_ptrs.len() as i32, opt_ptrs.as_ptr());
            if rc != 0 {
                let mut n = 0usize;
                (f.nvrtcGetProgramLogSize)(prog, &mut n);
                let mut log = vec![0 as c_char; n.max(1)];
                (f.nvrtcGetProgramLog)(prog, log.as_mut_ptr());
                (f.nvrtcDestroyProgram)(&mut prog);
                return Err(format!("kernel compile failed:\n{}", CStr::from_ptr(log.as_ptr()).to_string_lossy()));
            }
            let mut n = 0usize;
            (f.nvrtcGetCUBINSize)(prog, &mut n);
            let mut bin = vec![0u8; n];
            (f.nvrtcGetCUBIN)(prog, bin.as_mut_ptr() as *mut c_char);
            (f.nvrtcDestroyProgram)(&mut prog);
            bin
        };
        if let (true, Some(dir), Some(file)) = (use_cache, &dir, &file) {
            // best effort: a missing cache only costs the next start a compile
            if std::fs::create_dir_all(dir).is_ok() {
                let tmp = file.with_extension("tmp");
                if std::fs::write(&tmp, &bin).is_ok() {
                    let _ = std::fs::rename(&tmp, file);
                }
            }
        }
        Ok(bin)
    }

    /// The kernels, on a stream of their own (`private`: see Kernels::private).
    fn compile_on(private: bool) -> Result<Kernels, String> {
        let a = api()?;
        let f = &a.f;
        let bin = Self::binary(a)?;
        unsafe {
            let mut module: Ptr = std::ptr::null_mut();
            a.check((f.cuModuleLoadData)(&mut module, bin.as_ptr() as *const c_void), "load kernels")?;
            let get = |name: &str| -> Result<Ptr, String> {
                let c = CString::new(name).unwrap();
                let mut func: Ptr = std::ptr::null_mut();
                a.check((f.cuModuleGetFunction)(&mut func, module, c.as_ptr()), name)?;
                Ok(func)
            };
            // GHOST_INFER_KERNEL_INFO=1: registers and spills of the decode and verify kernels, for tuning
            if std::env::var_os("GHOST_INFER_KERNEL_INFO").is_some() {
                for name in ["attention_partial_gqa", "matmul_q4k_dp4a", "matmul_q4k_dp4a8", "matmul_q4k_vfy1", "matmul_q4k_vfy2", "matmul_q4k_vfy4", "matmul_q6k_vfy1", "matmul_q6k_vfy2", "matmul_q6k_vfy4"] {
                    let f = get(name)?;
                    let (mut regs, mut local) = (0, 0);
                    (a.f.cuFuncGetAttribute)(&mut regs, 4, f);
                    (a.f.cuFuncGetAttribute)(&mut local, 3, f);
                    eprintln!("[{name}: {regs} registers, {local} bytes of local memory per thread]");
                }
            }
            Ok(Kernels {
                embed: get("embed")?,
                rmsnorm: get("rmsnorm")?,
                rmsnorm_q8: get("rmsnorm_q8")?,
                qk_norm_rope: get("qk_norm_rope")?,
                matmul_q8: get("matmul_q8")?,
                matmul_q4k: [get("matmul_q4k_1")?, get("matmul_q4k_4")?],
                matmul_q6k: [get("matmul_q6k_1")?, get("matmul_q6k_4")?],
                matmul_dp4a: [
                    [get("matmul_q8_dp4a")?, get("matmul_q8_dp4a8")?],
                    [get("matmul_q4k_dp4a")?, get("matmul_q4k_dp4a8")?],
                    [get("matmul_q6k_dp4a")?, get("matmul_q6k_dp4a8")?],
                ],
                argmax_rows: get("argmax_rows")?,
                csm_sample: get("csm_sample")?,
                csm_embed: get("csm_embed")?,
                wh_layernorm: get("wh_layernorm")?,
                wh_bias: get("wh_bias")?,
                wh_add: get("wh_add")?,
                wh_im2col: get("wh_im2col")?,
                wh_attention: get("wh_attention")?,
                matmul_tiled: get("matmul_tiled")?,
                matmul_mma: if a.compute >= (7, 5) && std::env::var_os("GHOST_INFER_NO_MMA").is_none() { Some(get("matmul_mma")?) } else { None },
                // GHOST_INFER_NO_Q8K=1 keeps K-quant prompts on matmul_mma (per-32 activation scales)
                matmul_mma_k: if a.compute >= (7, 5) && std::env::var_os("GHOST_INFER_NO_MMA").is_none() && std::env::var_os("GHOST_INFER_NO_Q8K").is_none() {
                    Some([get("matmul_mma_q4k")?, get("matmul_mma_q6k")?])
                } else {
                    None
                },
                // GHOST_INFER_NO_VFY=1: verify batches stay on the dp4a kernels (8 tokens at most)
                matmul_vfy: if a.compute >= (7, 5) && std::env::var_os("GHOST_INFER_NO_VFY").is_none() {
                    // they live on shared memory: ask for the largest carveout, or only one block fits per SM
                    const PREFERRED_SHARED_CARVEOUT: i32 = 9;
                    let vfy = |name: &str| get(name).inspect(|&f| {
                        (a.f.cuFuncSetAttribute)(f, PREFERRED_SHARED_CARVEOUT, 100);
                    });
                    Some([
                        [vfy("matmul_q4k_vfy1")?, vfy("matmul_q4k_vfy2")?, vfy("matmul_q4k_vfy4")?],
                        [vfy("matmul_q6k_vfy1")?, vfy("matmul_q6k_vfy2")?, vfy("matmul_q6k_vfy4")?],
                    ])
                } else {
                    None
                },
                quantize_q8: get("quantize_q8")?,
                quantize_q8k: get("quantize_q8k")?,
                attention_partial: get("attention_partial")?,
                attention_partial_gqa: get("attention_partial_gqa")?,
                // GHOST_INFER_NO_VFY_ATT=1: verify batches attend token by token (attention_partial_gqa)
                attention_vfy: if kv_int8() && std::env::var_os("GHOST_INFER_NO_VFY_ATT").is_none() {
                    Some([
                        [get("attention_vfy_s2")?, get("attention_vfy_o2")?],
                        [get("attention_vfy_s4")?, get("attention_vfy_o4")?],
                        [get("attention_vfy_s8")?, get("attention_vfy_o8")?],
                    ])
                } else {
                    None
                },
                attention_combine: get("attention_combine")?,
                attention_prefill: get("attention_prefill")?,
                // GHOST_INFER_NO_TC_ATT=1 keeps prompts on the f32 attention kernel
                attention_prefill_tc: if a.compute >= (7, 5) && std::env::var_os("GHOST_INFER_NO_TC_ATT").is_none() {
                    Some([get("attention_prefill_tc64")?, get("attention_prefill_tc128")?])
                } else {
                    None
                },
                attention_prefill_tc256: if a.compute >= (7, 5) && std::env::var_os("GHOST_INFER_NO_TC_ATT").is_none() { Some(get("attention_prefill_tc256")?) } else { None },
                attention_decode_tc256: if a.compute >= (7, 5) && kv_int8() && std::env::var_os("GHOST_INFER_NO_TC_ATT").is_none() && std::env::var_os("GHOST_INFER_NO_TC_DECODE").is_none() { Some(get("attention_decode_tc256")?) } else { None },
                silu_mul: get("silu_mul")?,
                silu_q8: get("silu_q8")?,
                add_rows: get("add_rows")?,
                moe_gate_up_q3k: get("moe_gate_up_q3k")?,
                moe_down_q4k: get("moe_down_q4k")?,
                moe_combine: get("moe_combine")?,
                router_logits: get("router_logits")?,
                router_logits1: get("router_logits1")?,
                moe_gather: get("moe_gather")?,
                moe_scatter: get("moe_scatter")?,
                q3k_to_q8: get("q3k_to_q8")?,
                q5k_to_q8: get("q5k_to_q8")?,
                rmsnorm_rows: get("rmsnorm_rows")?,
                mla_kv: get("mla_kv")?,
                mla_q_rope: get("mla_q_rope")?,
                head_matvec: get("head_matvec")?,
                mla_attention: get("mla_attention")?,
                mla_combine: get("mla_combine")?,
                next_split_qgate: get("next_split_qgate")?,
                attention_wide: get("attention_wide")?,
                attention_wide_g: get("attention_wide_g")?,
                sigmoid_mul: get("sigmoid_mul")?,
                delta_conv: get("delta_conv")?,
                delta_prep: get("delta_prep")?,
                delta_scan: get("delta_scan")?,
                delta_norm: get("delta_norm")?,
                gated_add: get("gated_add")?,
                moe_gate_up_iq2: get("moe_gate_up_iq2")?,
                moe_down_iq2: get("moe_down_iq2")?,
                iq2_to_q8: get("iq2_to_q8")?,
                matmul_iq2_dp4a: get("matmul_iq2_dp4a")?,
                matmul_q3k_dp4a: [get("matmul_q3k_dp4a")?, get("matmul_q3k_dp4a8")?],
                stream: {
                    let mut st: Ptr = std::ptr::null_mut();
                    const NON_BLOCKING: u32 = 1;
                    a.check((f.cuStreamCreate)(&mut st, if private { NON_BLOCKING } else { 0 }), "stream")?;
                    st
                },
                private,
            })
        }
    }

    /// Copies `data` to device address `dst`: in this stream's order (private), or synchronously.
    fn h2d<T: Copy>(&self, dst: u64, data: &[T]) -> Result<(), String> {
        let a = api()?;
        let bytes = std::mem::size_of_val(data);
        if self.private {
            // from pageable memory: returns once the data is staged, so it may be dropped then
            a.check(unsafe { (a.f.cuMemcpyHtoDAsync_v2)(dst, data.as_ptr() as *const c_void, bytes, self.stream) }, "copy to GPU")
        } else {
            a.check(unsafe { (a.f.cuMemcpyHtoD_v2)(dst, data.as_ptr() as *const c_void, bytes) }, "copy to GPU")
        }
    }

    /// Copies from device address `src` into `out`, after the work queued before it.
    fn d2h<T: Copy>(&self, out: &mut [T], src: u64) -> Result<(), String> {
        let a = api()?;
        let bytes = std::mem::size_of_val(out);
        if self.private {
            // to pageable memory: returns once the copy is done
            a.check(unsafe { (a.f.cuMemcpyDtoHAsync_v2)(out.as_mut_ptr() as *mut c_void, src, bytes, self.stream) }, "copy from GPU")
        } else {
            a.check(unsafe { (a.f.cuMemcpyDtoH_v2)(out.as_mut_ptr() as *mut c_void, src, bytes) }, "copy from GPU")
        }
    }

    fn launch(&self, func: Ptr, grid: (u32, u32), block: u32, shared: u32, args: &[Arg]) -> Result<(), String> {
        self.launch3(func, (grid.0, grid.1, 1), block, shared, args)
    }

    fn launch3(&self, func: Ptr, grid: (u32, u32, u32), block: u32, shared: u32, args: &[Arg]) -> Result<(), String> {
        let a = api()?;
        // each argument lives in a u64 slot; the kernel reads the first 4 or 8 bytes (little-endian)
        let mut slots: Vec<u64> = args
            .iter()
            .map(|arg| match *arg {
                Arg::P(p) => p,
                Arg::I(i) => i as u32 as u64,
                Arg::F(f) => f.to_bits() as u64,
            })
            .collect();
        let mut params: Vec<Ptr> = slots.iter_mut().map(|s| s as *mut u64 as Ptr).collect();
        a.check(
            unsafe { (a.f.cuLaunchKernel)(func, grid.0, grid.1, grid.2, block, 1, 1, shared, self.stream, params.as_mut_ptr(), std::ptr::null_mut()) },
            "kernel launch",
        )
    }
}

// ---------------------------------------------------------------- weight upload

/// Streams host data into device memory through two pinned staging buffers on a copy stream of its own:
/// while the GPU copies one buffer, the CPU fills the other (several threads, so reading the mapped
/// model file overlaps too). Device memory is ready once `finish` returns.
struct Uploader {
    pinned: [*mut u8; 2],
    events: [Ptr; 2],
    stream: Ptr,
    next: usize,
    files: Vec<std::fs::File>, // one read handle per worker thread
    wait_secs: f64,            // GHOST_INFER_LOAD_TIMES: time spent waiting for the GPU copies
    file_secs: f64,            // reading (and repacking) tensors
    alloc_secs: f64,
    bytes: usize,
    arena: Option<Arena>, // where alloc places buffers (None: each its own cuMemAlloc)
}

const STAGE_BYTES: usize = 32 << 20;

/// Fills `buf` from `file` at `offset` (positioned reads: no shared cursor).
fn read_exact_at(file: &std::fs::File, mut buf: &mut [u8], mut offset: u64) -> Result<(), String> {
    #[cfg(windows)]
    use std::os::windows::fs::FileExt;
    #[cfg(not(windows))]
    use std::os::unix::fs::FileExt;
    while !buf.is_empty() {
        #[cfg(windows)]
        let n = file.seek_read(buf, offset);
        #[cfg(not(windows))]
        let n = file.read_at(buf, offset);
        match n {
            Ok(0) => return Err("model file ended early".into()),
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) => return Err(format!("reading the model file: {e}")),
        }
    }
    Ok(())
}
const GPU_Q6K_BYTES: usize = 208; // a Q6_K block on the GPU, without its d (must match Q6K_BYTES in the kernels)
const Q3K_GPU_BYTES: usize = 112; // a Q3_K block repacked for the GPU (must match Q3K_GPU_BYTES in the kernels)
/// Values of 3-bit weights converted to Q8 at a time for a prompt batch (16 MB of scratch).
const Q3K_CONV: usize = 16 << 20;

impl Uploader {
    fn new() -> Result<Uploader, String> {
        let a = api()?;
        let mut up = Uploader { pinned: [std::ptr::null_mut(); 2], events: [std::ptr::null_mut(); 2], stream: std::ptr::null_mut(), next: 0, files: Vec::new(), wait_secs: 0.0, bytes: 0, file_secs: 0.0, alloc_secs: 0.0, arena: None };
        unsafe {
            for i in 0..2 {
                let mut p: *mut c_void = std::ptr::null_mut();
                a.check((a.f.cuMemAllocHost_v2)(&mut p, STAGE_BYTES), "pinned staging buffer")?;
                up.pinned[i] = p as *mut u8;
                a.check((a.f.cuEventCreate)(&mut up.events[i], 0), "event")?;
            }
            a.check((a.f.cuStreamCreate)(&mut up.stream, 0), "copy stream")?;
        }
        Ok(up)
    }

    /// Copies `src` to device address `dst`.
    fn put(&mut self, dst: u64, src: &[u8]) -> Result<(), String> {
        use rayon::prelude::*;
        let a = api()?;
        for (i, chunk) in src.chunks(STAGE_BYTES).enumerate() {
            let b = self.next;
            self.next ^= 1;
            unsafe {
                let tw = std::time::Instant::now();
                a.check((a.f.cuEventSynchronize)(self.events[b]), "staging wait")?; // the GPU is done with this buffer
                self.wait_secs += tw.elapsed().as_secs_f64();
                self.bytes += chunk.len();
                let stage = std::slice::from_raw_parts_mut(self.pinned[b], chunk.len());
                stage.par_chunks_mut(1 << 20).zip(chunk.par_chunks(1 << 20)).for_each(|(d, s)| d.copy_from_slice(s));
                a.check((a.f.cuMemcpyHtoDAsync_v2)(dst + (i * STAGE_BYTES) as u64, self.pinned[b] as *const c_void, chunk.len(), self.stream), "copy to GPU")?;
                a.check((a.f.cuEventRecord)(self.events[b], self.stream), "event record")?;
            }
        }
        Ok(())
    }

    /// Copies `len` bytes of the file at `path`, from `offset`, to device address `dst`: read straight
    /// into the staging buffers by several threads, each with its own handle (Windows serialises
    /// synchronous reads on one handle), which is faster than paging the memory-mapped file in.
    fn put_file(&mut self, dst: u64, path: &std::path::Path, offset: u64, len: usize) -> Result<(), String> {
        use rayon::prelude::*;
        let a = api()?;
        self.open(path)?;
        let parts = self.files.len();
        let mut done = 0;
        while done < len {
            let n = (len - done).min(STAGE_BYTES);
            let b = self.next;
            self.next ^= 1;
            unsafe {
                let tw = std::time::Instant::now();
                a.check((a.f.cuEventSynchronize)(self.events[b]), "staging wait")?;
                self.wait_secs += tw.elapsed().as_secs_f64();
                self.bytes += n;
                let stage = std::slice::from_raw_parts_mut(self.pinned[b], n);
                let piece = n.div_ceil(parts);
                let files = &self.files;
                stage.par_chunks_mut(piece).enumerate().try_for_each(|(i, buf)| read_exact_at(&files[i], buf, offset + (done + i * piece) as u64))?;
                a.check((a.f.cuMemcpyHtoDAsync_v2)(dst + done as u64, self.pinned[b] as *const c_void, n, self.stream), "copy to GPU")?;
                a.check((a.f.cuEventRecord)(self.events[b], self.stream), "event record")?;
            }
            done += n;
        }
        Ok(())
    }

    fn open(&mut self, path: &std::path::Path) -> Result<(), String> {
        if self.files.is_empty() {
            for _ in 0..rayon::current_num_threads().max(1) {
                self.files.push(std::fs::File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?);
            }
        }
        Ok(())
    }

    /// A tensor of lk_in-byte blocks from the file, rewritten block by block (split: the
    /// lk_out-byte GPU block, returning the block's scale for d) straight into the staging buffers
    /// and copied to dst. Reading and repacking both run on all threads.
    fn put_repacked(
        &mut self,
        dst: u64,
        path: &std::path::Path,
        offset: u64,
        len: usize,
        (blk_in, blk_out): (usize, usize),
        d: &mut Vec<f32>,
        split: fn(&[u8], &mut [u8]) -> f32,
    ) -> Result<(), String> {
        use rayon::prelude::*;
        let a = api()?;
        self.open(path)?;
        let mut raw = vec![0u8; len];
        let piece = len.div_ceil(self.files.len());
        let files = &self.files;
        raw.par_chunks_mut(piece).enumerate().try_for_each(|(i, buf)| read_exact_at(&files[i], buf, offset + (i * piece) as u64))?;
        let blocks = len / blk_in;
        let per_stage = STAGE_BYTES / blk_out;
        d.reserve(blocks);
        for start in (0..blocks).step_by(per_stage) {
            let nb = per_stage.min(blocks - start);
            let b = self.next;
            self.next ^= 1;
            unsafe {
                let tw = std::time::Instant::now();
                a.check((a.f.cuEventSynchronize)(self.events[b]), "staging wait")?;
                self.wait_secs += tw.elapsed().as_secs_f64();
                self.bytes += nb * blk_out;
                let stage = std::slice::from_raw_parts_mut(self.pinned[b], nb * blk_out);
                let src = &raw[start * blk_in..(start + nb) * blk_in];
                let scales: Vec<f32> = stage.par_chunks_mut(blk_out).zip(src.par_chunks(blk_in)).map(|(o, i)| split(i, o)).collect();
                d.extend_from_slice(&scales);
                a.check((a.f.cuMemcpyHtoDAsync_v2)(dst + (start * blk_out) as u64, self.pinned[b] as *const c_void, nb * blk_out, self.stream), "copy to GPU")?;
                a.check((a.f.cuEventRecord)(self.events[b], self.stream), "event record")?;
            }
        }
        Ok(())
    }

    /// Device memory for an upload: in the arena if there is one.
    fn alloc(&mut self, bytes: usize) -> Result<DeviceBuf, String> {
        match &mut self.arena {
            Some(arena) => arena.alloc(bytes),
            None => DeviceBuf::alloc(bytes),
        }
    }

    fn put_vec<T: Copy>(&mut self, data: &[T]) -> Result<DeviceBuf, String> {
        let bytes = std::mem::size_of_val(data);
        let buf = self.alloc(bytes)?;
        self.put(buf.ptr, unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes) })?;
        Ok(buf)
    }

    fn finish(&mut self) -> Result<(), String> {
        let a = api()?;
        let tw = std::time::Instant::now();
        a.check(unsafe { (a.f.cuStreamSynchronize)(self.stream) }, "weight upload")?;
        self.wait_secs += tw.elapsed().as_secs_f64();
        if std::env::var_os("GHOST_INFER_LOAD_TIMES").is_some() {
            eprintln!(
                "[upload: {:.0} MB; reading and repacking {:.0} ms, allocations {:.0} ms, waiting for GPU copies {:.0} ms]",
                self.bytes as f64 / 1e6,
                self.file_secs * 1000.0,
                self.alloc_secs * 1000.0,
                self.wait_secs * 1000.0
            );
        }
        Ok(())
    }
}

impl Drop for Uploader {
    fn drop(&mut self) {
        if let Ok(a) = api() {
            unsafe {
                (a.f.cuStreamSynchronize)(self.stream);
                for i in 0..2 {
                    (a.f.cuMemFreeHost)(self.pinned[i] as *mut c_void);
                    (a.f.cuEventDestroy_v2)(self.events[i]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- model

/// Weight format on the GPU (matches `fmt` in the kernels).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fmt {
    /// int8 values + separate f32 scale per 32
    Q8 = 0,
    /// the GGUF blocks as is: 144 bytes per 256 (f16 d, f16 dmin, 12 scale bytes, 128 nibble bytes)
    Q4K = 1,
    /// 208-byte blocks of 256 (128 low-nibble bytes, 64 high-bit bytes, 16 int8 scales) + f32 d per block
    Q6K = 2,
    /// 2-bit i-quant blocks as in the file (GpuMatrix::iq says which); decoding reads them as they are, prompt
    /// batches convert them to Q8 first
    Iq2 = 3,
    /// Q3_K in 112-byte blocks (Q3K_GPU_BYTES); decoding reads them as they are, prompt batches convert them
    /// to Q8 a slice of rows at a time
    Q3K = 4,
}

/// A weight matrix on the GPU, repacked from its GGUF blocks so every field is naturally aligned and
/// every block starts on a 16-byte boundary.
struct GpuMatrix {
    fmt: Fmt,
    q: DeviceBuf,         // Q8: int8 values; K-quants: the blocks
    d: Option<DeviceBuf>, // Q8: scale per 32; Q6K: d per block
    rows: usize,
    cols: usize,
    /// Q8 converted from a 2-bit i-quant: decoding gives it int8 activations (no finer than its weights need)
    coarse: bool,
    /// Fmt::Iq2: the kernels' fmt (0 IQ2_XXS, 1 IQ2_S, 2 IQ2_XS)
    iq: i32,
}

impl GpuMatrix {
    fn d_ptr(&self) -> u64 {
        self.d.as_ref().map_or(0, |d| d.ptr)
    }

    /// A matrix in memory owned elsewhere (an expert slot, a scratch buffer); never freed through this.
    fn view(fmt: Fmt, q: u64, d: Option<u64>, rows: usize, cols: usize) -> std::mem::ManuallyDrop<GpuMatrix> {
        std::mem::ManuallyDrop::new(GpuMatrix { fmt, q: DeviceBuf { ptr: q, bytes: 0, owned: false }, d: d.map(|ptr| DeviceBuf { ptr, bytes: 0, owned: false }), rows, cols, coarse: false, iq: -1 })
    }
}

/// The kernels' fmt for 2-bit i-quant weights kept as they are (Fmt::Iq2).
fn iq2_kernel_fmt(ty: GgmlType) -> Option<i32> {
    match ty {
        GgmlType::Iq2Xxs => Some(0),
        GgmlType::Iq2S => Some(1),
        GgmlType::Iq2Xs => Some(2),
        _ => None,
    }
}

/// Uploads tensors whose rows are concatenated in one output (e.g. Q, K and V). Neighbours of the same
/// format are merged into one matrix (one launch); returns each matrix with its output row offset.
fn upload_parts(g: &Gguf, names: &[String], up: &mut Uploader) -> Result<Vec<(GpuMatrix, usize)>, String> {
    upload_parts_as(g, names, up, true)
}

/// `upload_parts`; `keep_iq2`: 2-bit i-quants stay as they are (Fmt::Iq2), else they become Q8 like the
/// rarer formats (the embedding table: its lookup kernel reads Q8 and K-quants only).
fn upload_parts_as(g: &Gguf, names: &[String], up: &mut Uploader, keep_iq2: bool) -> Result<Vec<(GpuMatrix, usize)>, String> {
    let mut parts = Vec::new();
    let mut i = 0;
    let mut row_off = 0;
    while i < names.len() {
        let ty = g.tensor(&names[i]).ok_or_else(|| format!("missing tensor {}", names[i]))?.ty;
        // the run of neighbours with this format, merged into one matrix
        let mut run = Vec::new();
        while let Some(t) = names.get(i).and_then(|n| g.tensor(n)).filter(|t| t.ty == ty) {
            let cols = t.dims[0] as usize;
            if (cols % 256 != 0 && ty.is_kquant()) || cols % 32 != 0 {
                return Err(format!("unsupported row length {cols} in {}", names[i]));
            }
            run.push(t);
            i += 1;
        }
        let cols = run[0].dims[0] as usize;
        let rows: usize = run.iter().map(|t| t.dims[1] as usize).sum();
        let iq = iq2_kernel_fmt(ty).filter(|_| keep_iq2 && cols % 256 == 0);
        let (fmt, gpu_bytes) = match ty {
            _ if iq.is_some() => (Fmt::Iq2, run.iter().map(|t| t.n_bytes).sum()),
            GgmlType::Q3K if keep_iq2 => (Fmt::Q3K, rows * (cols / 256) * Q3K_GPU_BYTES),
            // no 5-bit GPU kernels yet: requantised to int8 (about 1.5x the memory), as are the rarer formats
            // (Q5_0, IQ4_NL, ...) and Q3_K in the embedding table
            GgmlType::Q8_0 | GgmlType::Q3K | GgmlType::Q5K => (Fmt::Q8, rows * cols),
            t if t.dequant_only() => (Fmt::Q8, rows * cols),
            GgmlType::Q4K => (Fmt::Q4K, run.iter().map(|t| t.n_bytes).sum()),
            GgmlType::Q6K => (Fmt::Q6K, rows * (cols / 256) * GPU_Q6K_BYTES),
            other => return Err(format!("GPU backend does not handle {} weights yet", other.name())),
        };
        let ta = std::time::Instant::now();
        let q = up.alloc(gpu_bytes)?;
        up.alloc_secs += ta.elapsed().as_secs_f64();
        let mut off = 0;
        let mut d = Vec::new();
        for t in run {
            let tf = std::time::Instant::now();
            let dst = q.ptr + off as u64;
            match fmt {
                // the GGUF blocks as they are, straight from the file
                Fmt::Q4K | Fmt::Iq2 => up.put_file(dst, &g.path, t.offset as u64, t.n_bytes)?,
                // 210-byte blocks: the f16 d at the end goes to its own array
                Fmt::Q6K => up.put_repacked(dst, &g.path, t.offset as u64, t.n_bytes, (210, GPU_Q6K_BYTES), &mut d, |i, o| {
                    o.copy_from_slice(&i[..GPU_Q6K_BYTES]);
                    f16_to_f32(u16::from_le_bytes([i[GPU_Q6K_BYTES], i[GPU_Q6K_BYTES + 1]]))
                })?,
                // 110-byte blocks reordered: header (scales, d) first, then the high bits and the 2-bit pairs
                Fmt::Q3K => up.put_repacked(dst, &g.path, t.offset as u64, t.n_bytes, (110, Q3K_GPU_BYTES), &mut d, |i, o| {
                    q3k_repack(i, o);
                    0.0
                })?,
                Fmt::Q8 if t.ty != GgmlType::Q8_0 => {
                    let (qb, ds) = requant_q8(g, t);
                    up.put(dst, &qb)?;
                    d.extend_from_slice(&ds);
                }
                // 34-byte blocks: f16 d, then 32 int8 values
                Fmt::Q8 => up.put_repacked(dst, &g.path, t.offset as u64, t.n_bytes, (34, 32), &mut d, |i, o| {
                    o.copy_from_slice(&i[2..]);
                    f16_to_f32(u16::from_le_bytes([i[0], i[1]]))
                })?,
            }
            off += match fmt {
                Fmt::Q4K | Fmt::Iq2 => t.n_bytes,
                Fmt::Q6K => t.n_bytes / 210 * GPU_Q6K_BYTES,
                Fmt::Q3K => t.n_bytes / 110 * Q3K_GPU_BYTES,
                Fmt::Q8 => t.n_elements() as usize,
            };
            up.file_secs += tf.elapsed().as_secs_f64();
        }
        let d = if !matches!(fmt, Fmt::Q4K | Fmt::Iq2 | Fmt::Q3K) { Some(up.put_vec(&d)?) } else { None };
        // requantised 3- and 5-bit weights decode with int8 activations like 2-bit ones (Q3_K_M keeps a few
        // Q5_K matrices beside its Q3_K ones: else those layers' decoding would leave the dp4a path)
        let coarse = ty.is_iquant() || matches!(ty, GgmlType::Q3K | GgmlType::Q5K);
        parts.push((GpuMatrix { fmt, q, d, rows, cols, coarse, iq: iq.unwrap_or(-1) }, row_off));
        row_off += rows;
    }
    Ok(parts)
}

/// A 2-D tensor of another format (Q3_K, Q5_K) as GPU Q8: int8 values and an f32 scale per 32,
/// requantised from the dequantised rows on every CPU thread.
fn requant_q8(g: &Gguf, t: &TensorInfo) -> (Vec<u8>, Vec<f32>) {
    use rayon::prelude::*;
    let (cols, rows) = (t.dims[0] as usize, t.dims[1] as usize);
    let m = crate::quant::Matrix { data: &g.mmap[t.offset..t.offset + t.n_bytes], ty: t.ty, rows, cols };
    let mut q = vec![0u8; rows * cols];
    let mut d = vec![0f32; rows * cols / 32];
    q.par_chunks_mut(cols).zip(d.par_chunks_mut(cols / 32)).enumerate().for_each_init(
        || vec![0f32; cols],
        |row, (r, (qr, dr))| {
            m.row_f32(r, row);
            for ((blk, qb), dv) in row.chunks_exact(32).zip(qr.chunks_exact_mut(32)).zip(dr) {
                let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
                let s = amax / 127.0;
                let inv = if s != 0.0 { 1.0 / s } else { 0.0 };
                for (o, v) in qb.iter_mut().zip(blk) {
                    *o = (v * inv).round() as i8 as u8;
                }
                *dv = s;
            }
        },
    );
    (q, d)
}

/// A GGUF Q3_K block (110 bytes) as the GPU's 112: scales, d and 2 pad bytes, then the high bits and the 2-bit pairs.
fn q3k_repack(blk: &[u8], o: &mut [u8]) {
    o[..12].copy_from_slice(&blk[96..108]); // scales
    o[12..14].copy_from_slice(&blk[108..110]); // d
    o[14..16].fill(0);
    o[16..48].copy_from_slice(&blk[..32]); // high bits
    o[48..112].copy_from_slice(&blk[32..96]); // 2-bit pairs
}

fn upload_one(g: &Gguf, name: &str, up: &mut Uploader) -> Result<GpuMatrix, String> {
    Ok(upload_parts(g, &[name.to_string()], up)?.pop().unwrap().0)
}

/// A tensor as f16 rows (any format; a 3-D one's matrices one after the other): the latent attention's
/// small per-head projections.
fn upload_f16(g: &Gguf, name: &str, up: &mut Uploader) -> Result<DeviceBuf, String> {
    let t = g.tensor(name).ok_or_else(|| format!("missing tensor {name}"))?;
    let cols = t.dims[0] as usize;
    let rows = t.n_elements() as usize / cols;
    let m = crate::quant::Matrix { data: &g.mmap[t.offset..t.offset + t.n_bytes], ty: t.ty, rows, cols };
    let mut h = vec![0u16; rows * cols];
    let mut row = vec![0f32; cols];
    for (r, out) in h.chunks_exact_mut(cols).enumerate() {
        m.row_f32(r, &mut row);
        for (o, v) in out.iter_mut().zip(&row) {
            *o = crate::quant::f32_to_f16(*v);
        }
    }
    up.put_vec(&h)
}

/// A vector of any format as f32 (qwen3next's BF16 shared-expert gate).
fn upload_f32(g: &Gguf, name: &str, up: &mut Uploader) -> Result<DeviceBuf, String> {
    let t = g.tensor(name).ok_or_else(|| format!("missing tensor {name}"))?;
    let n = t.n_elements() as usize;
    let mut v = vec![0f32; n];
    crate::quant::Matrix { data: &g.mmap[t.offset..t.offset + t.n_bytes], ty: t.ty, rows: 1, cols: n }.row_f32(0, &mut v);
    up.put_vec(&v)
}

fn upload_vec(g: &Gguf, name: &str, up: &mut Uploader) -> Result<DeviceBuf, String> {
    let t = g.tensor(name).ok_or_else(|| format!("missing tensor {name}"))?;
    if t.ty != GgmlType::F32 {
        return Err(format!("GPU backend expects F32 norm weights ({name} is {})", t.ty.name()));
    }
    let buf = up.alloc(t.n_bytes)?;
    up.put(buf.ptr, &g.mmap[t.offset..t.offset + t.n_bytes])?;
    Ok(buf)
}

struct GpuLayer {
    attn_norm: DeviceBuf,
    qkv: Vec<(GpuMatrix, usize)>, // with output row offsets in the packed [q | k | v] rows (empty: latent attention)
    q_norm: Option<DeviceBuf>,    // Qwen3 per-head norms
    k_norm: Option<DeviceBuf>,
    mla: Option<GpuMla>,
    delta: Option<GpuDelta>,
    wo: GpuMatrix, // a DeltaNet layer's ssm_out
    ffn_norm: DeviceBuf,
    ffn: GpuFfn,
}

/// A Gated DeltaNet layer (see model::Delta): one projection to [q k v | z | ba] rows, the convolution taps,
/// dt_bias, A (-exp(A_log)) and the output norm.
struct GpuDelta {
    proj: Vec<(GpuMatrix, usize)>,
    conv: DeviceBuf,
    dt_bias: DeviceBuf,
    a: DeviceBuf,
    norm: DeviceBuf,
}

/// Latent attention weights (see model::Mla): the low-rank query step and its norm (if any), the query
/// projection, the compressed key-value projection and its norm, and the per-head k_b / v_b as f16.
struct GpuMla {
    q_a: Option<(GpuMatrix, DeviceBuf)>,
    q: GpuMatrix,
    kv_a: GpuMatrix,
    kv_norm: DeviceBuf,
    k_b: DeviceBuf,
    v_b: DeviceBuf,
}

enum GpuFfn {
    Dense { gate_up: Vec<(GpuMatrix, usize)>, down: GpuMatrix }, // gate_up: [gate | up]
    /// mixture of experts: routed and computed on the CPU (GpuModel::moe); deepseek2 and qwen3next add a
    /// shared expert, as a dense block on the GPU, which qwen3next scales by sigmoid(gate · x)
    Moe { shared: Option<(Vec<(GpuMatrix, usize)>, GpuMatrix)>, gate: Option<DeviceBuf> },
}

/// What the current prompt pass is doing, for the server's /api/progress (the chat page shows it while it
/// waits for the first token): prompt tokens read of the total, and time spent waiting for expert weights
/// to come off the disk (a mixture of experts bigger than the free RAM).
pub struct Progress {
    done: AtomicUsize,
    total: AtomicUsize, // 0: no prompt pass running
    base: AtomicUsize,  // tokens of the earlier batches of this prompt
    disk_ms: AtomicU64,
    disk_at: AtomicU64,  // clock_ms() of the last wait
    on_disk: AtomicBool, // waiting for the prefetch thread's read right now
}

pub static PROGRESS: Progress = Progress {
    done: AtomicUsize::new(0),
    total: AtomicUsize::new(0),
    base: AtomicUsize::new(0),
    disk_ms: AtomicU64::new(0),
    disk_at: AtomicU64::new(0),
    on_disk: AtomicBool::new(false),
};

fn clock_ms() -> u64 {
    static T0: OnceLock<std::time::Instant> = OnceLock::new();
    T0.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64 + 1
}

impl Progress {
    /// (tokens read, prompt tokens (0: none running), ms waited on the disk in this prompt, waited in the last second)
    pub fn get(&self) -> (usize, usize, u64, bool) {
        let at = self.disk_at.load(Ordering::Relaxed);
        let disk = self.on_disk.load(Ordering::Relaxed) || (at > 0 && clock_ms() - at < 1000);
        (self.done.load(Ordering::Relaxed), self.total.load(Ordering::Relaxed), self.disk_ms.load(Ordering::Relaxed), disk)
    }
    fn start(&self, total: usize) {
        self.done.store(0, Ordering::Relaxed);
        self.base.store(0, Ordering::Relaxed);
        self.disk_ms.store(0, Ordering::Relaxed);
        self.total.store(total, Ordering::Relaxed);
    }
    /// Layer l of n_layers is done for the current batch of n tokens.
    fn layer(&self, l: usize, n_layers: usize, n: usize) {
        self.done.store(self.base.load(Ordering::Relaxed) + n * (l + 1) / n_layers, Ordering::Relaxed);
    }
    fn waited_on_disk(&self, ms: u64) {
        self.disk_ms.fetch_add(ms, Ordering::Relaxed);
        self.disk_at.store(clock_ms(), Ordering::Relaxed);
    }
}

/// Reads expert weights in ahead of use, on a thread of its own: Windows is asked for each request's ranges
/// as large concurrent reads, then every page is touched so they are in memory when the experts run. On a
/// hard disk this is several times faster than the 4 KB page faults the experts would otherwise take.
struct Prefetcher {
    tx: Option<std::sync::mpsc::Sender<Vec<(usize, usize)>>>,
    thread: Option<std::thread::JoinHandle<()>>,
    sent: std::sync::Mutex<(u64, Vec<(usize, u64)>)>, // requests so far, and (layer, request number) not yet waited for
    done: Arc<(std::sync::Mutex<u64>, std::sync::Condvar)>, // requests finished
}

impl Prefetcher {
    fn new() -> Prefetcher {
        let (tx, rx) = std::sync::mpsc::channel::<Vec<(usize, usize)>>();
        let done = Arc::new((std::sync::Mutex::new(0u64), std::sync::Condvar::new()));
        let finished = done.clone();
        let thread = std::thread::Builder::new()
            .name("expert prefetch".into())
            .spawn(move || {
                let log = std::env::var_os("GHOST_INFER_LOAD_TIMES").is_some();
                let mark = || {
                    *finished.0.lock().unwrap() += 1;
                    finished.1.notify_all();
                };
                for ranges in rx {
                    let t = std::time::Instant::now();
                    os::prefetch(&ranges);
                    let asked = t.elapsed();
                    for &(at, n) in &ranges {
                        for p in (at..at + n).step_by(4096) {
                            unsafe { std::ptr::read_volatile(p as *const u8) };
                        }
                    }
                    mark();
                    if log {
                        let mb = ranges.iter().map(|r| r.1).sum::<usize>() as f64 / 1e6;
                        eprintln!("[prefetch {mb:.0} MB: asked in {:.0} ms, in memory after {:.0} ms]", asked.as_secs_f64() * 1e3, t.elapsed().as_secs_f64() * 1e3);
                    }
                }
            })
            .ok();
        Prefetcher { tx: Some(tx), thread, sent: std::sync::Mutex::new((0, Vec::new())), done }
    }

    /// Queues the ranges of layer l.
    fn request(&self, l: usize, ranges: Vec<(usize, usize)>) {
        let (Some(tx), Some(_)) = (&self.tx, &self.thread) else { return };
        let mut sent = self.sent.lock().unwrap();
        if tx.send(ranges).is_ok() {
            sent.0 += 1;
            let n = sent.0;
            sent.1.retain(|&(pl, _)| pl != l);
            sent.1.push((l, n));
        }
    }

    /// Waits until layer l's latest request is in memory (none queued: returns at once); the time waited.
    fn wait(&self, l: usize) -> std::time::Duration {
        let t = std::time::Instant::now();
        let ticket = {
            let mut sent = self.sent.lock().unwrap();
            let at = sent.1.iter().position(|&(pl, _)| pl == l);
            at.map(|i| sent.1.remove(i).1)
        };
        if let Some(n) = ticket {
            let mut done = self.done.0.lock().unwrap();
            PROGRESS.on_disk.store(*done < n, Ordering::Relaxed);
            while *done < n {
                done = self.done.1.wait(done).unwrap();
            }
            PROGRESS.on_disk.store(false, Ordering::Relaxed);
        }
        t.elapsed()
    }
}

impl Drop for Prefetcher {
    fn drop(&mut self) {
        self.tx.take(); // ends the thread's loop once it has finished its current request
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The experts of a mixture-of-experts model, run on the CPU straight from the model file (its own
/// mapping of it; the OS shares the pages with any other).
struct CpuMoe {
    prefetch: Prefetcher, // first, so its thread has stopped before the mapping it reads goes
    mmap: memmap2::Mmap,
    n_experts: usize,
    routers: Vec<Vec<f32>>,          // per layer: n_experts rows of dim logit weights (empty: a dense layer)
    biases: Vec<Option<Vec<f32>>>,   // deepseek2: per-expert pick bias
    tensors: Vec<[ExpertTensor; 3]>, // per layer: gate, up, down (no rows: a dense layer)
}

/// Where one 3-D expert tensor lies in the file: expert e's matrix starts at offset + e * stride.
#[derive(Clone, Copy)]
struct ExpertTensor {
    offset: usize,
    stride: usize,
    ty: GgmlType,
    rows: usize,
    cols: usize,
}

impl CpuMoe {
    fn load(g: &Gguf, cfg: &Config) -> Result<CpuMoe, String> {
        let file = std::fs::File::open(&g.path).map_err(|e| format!("cannot open {}: {e}", g.path.display()))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("cannot map {}: {e}", g.path.display()))?;
        let mut routers = Vec::with_capacity(cfg.n_layers);
        let mut tensors = Vec::with_capacity(cfg.n_layers);
        let mut biases = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}.weight");
            if i < cfg.n_dense_lead {
                // deepseek2's dense first layers: run as dense blocks on the GPU, nothing here
                let none = ExpertTensor { offset: 0, stride: 0, ty: GgmlType::F32, rows: 0, cols: 0 };
                routers.push(Vec::new());
                biases.push(None);
                tensors.push([none; 3]);
                continue;
            }
            biases.push(match g.tensor(&format!("blk.{i}.exp_probs_b.bias")) {
                Some(t) if t.ty == GgmlType::F32 && t.n_elements() as usize == cfg.n_experts => Some(g.mmap[t.offset..t.offset + t.n_bytes].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()),
                Some(t) => return Err(format!("{}: expected {} F32 values", t.name, cfg.n_experts)),
                None => None,
            });
            let r = g.tensor(&p("ffn_gate_inp")).ok_or_else(|| format!("missing tensor {}", p("ffn_gate_inp")))?;
            if r.ty != GgmlType::F32 || r.n_elements() as usize != cfg.n_experts * cfg.dim {
                return Err(format!("{}: expected an F32 router of {} x {}", r.name, cfg.n_experts, cfg.dim));
            }
            routers.push(g.mmap[r.offset..r.offset + r.n_bytes].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect());
            let part = |name: &str, rows: usize, cols: usize| -> Result<ExpertTensor, String> {
                let t = g.tensor(&p(name)).ok_or_else(|| format!("missing tensor {}", p(name)))?;
                if t.ty.block().is_none() || t.dims[..] != [cols as u64, rows as u64, cfg.n_experts as u64] {
                    return Err(format!("{}: unexpected {} tensor of shape {:?}", t.name, t.ty.name(), t.dims));
                }
                Ok(ExpertTensor { offset: t.offset, stride: t.n_bytes / cfg.n_experts, ty: t.ty, rows, cols })
            };
            let (dim, ffn) = (cfg.dim, cfg.expert_ffn_dim);
            tensors.push([part("ffn_gate_exps", ffn, dim)?, part("ffn_up_exps", ffn, dim)?, part("ffn_down_exps", dim, ffn)?]);
        }
        Ok(CpuMoe { prefetch: Prefetcher::new(), mmap, n_experts: cfg.n_experts, routers, biases, tensors })
    }

    /// Starts reading layer l's experts (those `cached` does not hold in GPU memory) off the disk, as
    /// runs of neighbouring experts in file order.
    fn prefetch_layer(&self, l: usize, cached: &[i32]) {
        let base = self.mmap.as_ptr() as usize;
        let mut ranges = Vec::new();
        for t in &self.tensors[l] {
            let mut e = 0;
            while e < self.n_experts {
                if cached[e] >= 0 {
                    e += 1;
                    continue;
                }
                let first = e;
                while e < self.n_experts && cached[e] < 0 {
                    e += 1;
                }
                ranges.push((base + t.offset + first * t.stride, (e - first) * t.stride));
            }
        }
        if std::env::var_os("GHOST_INFER_NO_PREFETCH").is_none() {
            self.prefetch.request(l, ranges);
        }
    }

    fn expert(&self, l: usize, e: usize) -> crate::model::Expert<'_> {
        let m = |t: &ExpertTensor| {
            let at = t.offset + e * t.stride;
            crate::quant::Matrix { data: &self.mmap[at..at + t.stride], ty: t.ty, rows: t.rows, cols: t.cols }
        };
        let [gate, up, down] = &self.tensors[l];
        crate::model::Expert { gate: m(gate), up: m(up), down: m(down) }
    }
}

/// Host-side scratch of the CPU expert path.
#[derive(Default)]
struct MoeHost {
    xb: Vec<f32>,
    acc: Vec<f32>,
    picks: Vec<(usize, f32)>,
    cpu_picks: Vec<(usize, f32)>,
    scratch: crate::model::MoeScratch,
}

/// Experts queued for upload per token at most (GHOST_INFER_UPLOADS_PER_TOKEN): each moves ~2 MB through RAM
/// and PCIe in the background, bandwidth the CPU's experts share. Replaying routing traces of varied prompts
/// (Qwen3-30B-A3B, a third of its experts cached): 8 a token keep 72% of picks on the GPU, 16 80%, 24 84%,
/// every miss 88%; the cache before this one, which kept the most picked experts, 64%. Through the server
/// (scripts/bench-moe.py), median tokens/s: 8 30.5, 12 33.0, 16 32.8, 24 28.8.
fn uploads_per_token() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::env::var("GHOST_INFER_UPLOADS_PER_TOKEN").ok().and_then(|v| v.parse().ok()).unwrap_or(12))
}

/// An expert to copy into a cache slot: its gate, up and down in the mapping (address, bytes), the slot's
/// device address.
struct UploadJob {
    l: usize,
    e: usize,
    slot: usize,
    src: [(usize, usize); 3],
    gate_bytes: usize, // as the layer's experts sit in a slot (ExpertCache::pack_parts)
    raw: bool,
    dst: u64,
    evicted: Option<[(usize, usize); 3]>, // the expert the slot held, back to the CPU: its pages are read in
}

/// Copies experts into cache slots on a thread of its own: packs each into one of two pinned buffers and
/// copies it on a stream of its own, so the GPU's work and the CPU's experts go on meanwhile. Each slot is
/// reported back once its copy has landed.
struct ExpertUploader {
    tx: Option<std::sync::mpsc::Sender<UploadJob>>,
    done: std::sync::mpsc::Receiver<(usize, usize, usize)>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl ExpertUploader {
    fn new(slot_bytes: usize) -> ExpertUploader {
        use std::sync::mpsc::{channel, TryRecvError};
        let (tx, rx) = channel::<UploadJob>();
        let (done_tx, done) = channel();
        let thread = std::thread::Builder::new()
            .name("expert upload".into())
            .spawn(move || {
                let Ok(a) = api() else { return };
                let mut stream: Ptr = std::ptr::null_mut();
                let mut pinned = [std::ptr::null_mut::<c_void>(); 2];
                let mut events: [Ptr; 2] = [std::ptr::null_mut(); 2];
                let ok = unsafe {
                    (a.f.cuCtxSetCurrent)(a.ctx as Ptr) == 0
                        && (a.f.cuStreamCreate)(&mut stream, 1) == 0 // CU_STREAM_NON_BLOCKING
                        && (0..2).all(|i| (a.f.cuMemAllocHost_v2)(&mut pinned[i], slot_bytes) == 0 && (a.f.cuEventCreate)(&mut events[i], 2) == 0)
                };
                if !ok {
                    return; // no uploads: the CPU keeps running these experts
                }
                let mut inflight: [Option<((usize, usize, usize), [(usize, usize); 3])>; 2] = [None, None];
                let mut b = 0;
                // a landed copy: its slot is reported, and the expert's pages leave the working set (it runs
                // from GPU memory now; RAM is usually short of the whole model, and the CPU's experts need it)
                let land = |i: usize, inflight: &mut [Option<((usize, usize, usize), [(usize, usize); 3])>; 2]| {
                    if let Some((d, src)) = inflight[i].take() {
                        unsafe { (a.f.cuEventSynchronize)(events[i]) };
                        let _ = done_tx.send(d);
                        os::release(&src);
                    }
                };
                loop {
                    let job = match rx.try_recv() {
                        Ok(j) => j,
                        Err(TryRecvError::Disconnected) => {
                            // stopping (finish): the last copies are reported too
                            land(0, &mut inflight);
                            land(1, &mut inflight);
                            break;
                        }
                        Err(TryRecvError::Empty) => {
                            // nothing waiting: let the copies in flight land, then sleep until there is
                            land(0, &mut inflight);
                            land(1, &mut inflight);
                            match rx.recv() {
                                Ok(j) => j,
                                Err(_) => break,
                            }
                        }
                    };
                    land(b, &mut inflight); // this buffer's last copy is done with it
                    // SAFETY: the mapping outlives this thread (see GpuModel), and nothing else writes the buffer
                    let src = job.src.map(|(at, n)| unsafe { std::slice::from_raw_parts(at as *const u8, n) });
                    let buf = unsafe { std::slice::from_raw_parts_mut(pinned[b] as *mut u8, slot_bytes) };
                    let bytes = ExpertCache::pack_parts(src[0], src[1], src[2], job.gate_bytes, job.raw, buf);
                    unsafe {
                        (a.f.cuMemcpyHtoDAsync_v2)(job.dst, pinned[b], bytes, stream);
                        (a.f.cuEventRecord)(events[b], stream);
                    }
                    inflight[b] = Some(((job.l, job.e, job.slot), job.src));
                    b ^= 1;
                    // the expert this one replaces runs on the CPU from now on: fault its pages back into the
                    // working set here, while the copy runs, rather than in the middle of a token
                    if let Some(ranges) = job.evicted {
                        os::prefetch(&ranges);
                        for (at, n) in ranges {
                            for p in (at..at + n).step_by(4096) {
                                unsafe { std::ptr::read_volatile(p as *const u8) };
                            }
                        }
                    }
                }
                unsafe {
                    (a.f.cuStreamSynchronize)(stream);
                    for i in 0..2 {
                        (a.f.cuMemFreeHost)(pinned[i]);
                        (a.f.cuEventDestroy_v2)(events[i]);
                    }
                    (a.f.cuStreamDestroy_v2)(stream);
                }
            })
            .ok();
        ExpertUploader { tx: Some(tx), done, thread }
    }
}

impl ExpertUploader {
    /// The thread finishes the copies queued and stops; `done` still holds their slots.
    fn finish(&mut self) {
        self.tx = None;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for ExpertUploader {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Experts of a mixture-of-experts model held in GPU memory: an expert the CPU had to run is copied in (in
/// the background, up to `uploads_per_token`), over the one picked least recently. Consecutive tokens
/// reuse many experts, so recency predicts the next picks better than counts. Only layers whose experts
/// the GPU kernels handle (Q3_K gate and up, Q4_K down) are cached. Placement never changes which experts
/// run, only where (and the rounding of their sums).
struct ExpertCache {
    uploader: ExpertUploader, // first: its thread stops (copies landed) before the slots are freed
    blocks: Vec<(DeviceBuf, usize)>, // the slots' memory, SLOT_BLOCK slots a block: (buffer, its first slot)
    addr: Vec<u64>,                  // per slot: its device address
    max_slots: usize,                // one per expert of every layer
    slot_bytes: usize,
    /// per layer: its gate (= up) matrix as it sits in a slot (Q3_K repacked; 2-bit as in the file), down after both
    gate_bytes: Vec<usize>,
    slot_of: Vec<Vec<i32>>,             // [layer][expert] -> slot, or -1
    owner: Vec<Option<(usize, usize)>>, // per slot: (layer, expert)
    free: Vec<usize>,
    used: Vec<u64>,          // per slot: the layer step that last picked its expert (u64::MAX: being copied)
    clock: u64,              // layer steps so far
    queued: Vec<Vec<bool>>,  // [layer][expert]: on its way into a slot
    sent: usize,             // uploads queued this token
    cacheable: Vec<bool>,
    batch_ok: bool, // every layer's experts are formats the prompt-batch path converts (else batches run on the CPU)
    pub uploads: usize,
    // prompt batches: every expert runs on the GPU, the uncached ones staged through two pinned buffers
    // into two scratch slots (double-buffered: the next expert's copy overlaps this one's maths)
    routers: Vec<DeviceBuf>, // per layer, f32
    pinned: [*mut u8; 2],
    scratch: [DeviceBuf; 2],
    scratch_bytes: usize,
    events: [Ptr; 2],
    next: usize,
    conv_q: DeviceBuf, // an expert's [gate | up] (or a down) converted to Q8 for the batched kernels
    conv_d: DeviceBuf,
    idx: DeviceBuf, // per layer: the tokens of each expert in turn, and their weights
    staged: usize,
}

impl Drop for ExpertCache {
    fn drop(&mut self) {
        if let Ok(a) = api() {
            for p in self.pinned.iter().filter(|p| !p.is_null()) {
                unsafe { (a.f.cuMemFreeHost)(*p as *mut c_void) };
            }
        }
    }
}

impl ExpertCache {
    /// As many slots as `budget` bytes hold after the prompt-batch buffers (none if no layer is cacheable).
    fn new(moe: &CpuMoe, cfg: &Config, budget: usize) -> Result<ExpertCache, String> {
        let a = api()?;
        let q3k = |[g, u, _]: &[ExpertTensor; 3]| g.ty == GgmlType::Q3K && u.ty == GgmlType::Q3K;
        let cacheable: Vec<bool> = moe.tensors.iter().map(|t| (q3k(t) && t[2].ty == GgmlType::Q4K) || iq2_fmt(t).is_some()).collect();
        let batch_ok = moe.tensors.iter().filter(|t| t[0].rows > 0).all(|t| (q3k(t) && matches!(t[2].ty, GgmlType::Q4K | GgmlType::Q5K)) || iq2_fmt(t).is_some());
        moe.tensors.iter().find(|t| t[0].rows > 0).ok_or("no expert layers")?;
        let gate_bytes: Vec<usize> = moe.tensors.iter().map(|t| if q3k(t) { t[0].rows * (t[0].cols / 256) * Q3K_GPU_BYTES } else { t[0].stride }).collect();
        let expert_bytes = |l: usize| 2 * gate_bytes[l] + moe.tensors[l][2].stride;
        let slot_bytes = (0..moe.tensors.len()).filter(|&l| cacheable[l]).map(expert_bytes).max().unwrap_or(1);
        // a scratch slot holds any layer's expert (some downs are bigger formats than the cached ones)
        let scratch_bytes = (0..moe.tensors.len()).map(expert_bytes).max().unwrap_or(0);
        let g0 = moe.tensors.iter().find(|t| t[0].rows > 0).unwrap()[0];
        let routers = moe.routers.iter().map(|r| DeviceBuf::alloc(r.len().max(1) * 4).and_then(|b| b.write(r).map(|_| b))).collect::<Result<Vec<_>, _>>()?;
        let conv_elems = (2 * g0.rows * g0.cols).max(g0.rows * g0.cols);
        // the prompt-batch table: per token and pick, its token index and weight
        let idx_bytes = 8 * max_batch(cfg) * cfg.n_experts_used.max(8);
        let extras = routers.iter().map(|b| b.bytes).sum::<usize>() + 2 * scratch_bytes + conv_elems * 9 / 8 + idx_bytes;
        let max_slots = if cacheable.iter().any(|&c| c) { cfg.n_layers * cfg.n_experts } else { 0 };
        let mut pinned = [std::ptr::null_mut(); 2];
        let mut events = [std::ptr::null_mut(); 2];
        for i in 0..2 {
            let mut p: *mut c_void = std::ptr::null_mut();
            a.check(unsafe { (a.f.cuMemAllocHost_v2)(&mut p, scratch_bytes) }, "pinned expert staging")?;
            pinned[i] = p as *mut u8;
            a.check(unsafe { (a.f.cuEventCreate)(&mut events[i], 0) }, "event")?;
        }
        let mut cache = ExpertCache {
            uploader: ExpertUploader::new(slot_bytes),
            blocks: Vec::new(),
            addr: Vec::new(),
            max_slots,
            slot_bytes,
            gate_bytes,
            slot_of: vec![vec![-1; cfg.n_experts]; cfg.n_layers],
            owner: Vec::new(),
            free: Vec::new(),
            used: Vec::new(),
            clock: 0,
            queued: vec![vec![false; cfg.n_experts]; cfg.n_layers],
            sent: 0,
            cacheable,
            batch_ok,
            uploads: 0,
            routers,
            pinned,
            scratch: [DeviceBuf::alloc(scratch_bytes)?, DeviceBuf::alloc(scratch_bytes)?],
            scratch_bytes,
            events,
            next: 0,
            conv_q: DeviceBuf::alloc(conv_elems)?,
            conv_d: DeviceBuf::alloc(conv_elems / 32 * 4)?,
            idx: DeviceBuf::alloc(idx_bytes)?,
            staged: 0,
        };
        cache.add_blocks(budget.saturating_sub(extras));
        Ok(cache)
    }

    /// More slots, a block at a time, in up to `bytes` of GPU memory.
    fn add_blocks(&mut self, mut bytes: usize) {
        while self.slots() < self.max_slots {
            let n = (bytes / self.slot_bytes).min(SLOT_BLOCK).min(self.max_slots - self.slots());
            if n == 0 {
                break;
            }
            let Ok(buf) = DeviceBuf::alloc(n * self.slot_bytes) else { break };
            let first = self.slots();
            self.addr.extend((0..n).map(|i| buf.ptr + (i * self.slot_bytes) as u64));
            self.owner.resize(first + n, None);
            self.used.resize(first + n, 0);
            self.free.extend((first..first + n).rev());
            self.blocks.push((buf, first));
            bytes -= n * self.slot_bytes;
        }
    }

    /// Frees the last blocks of slots until `bytes` of GPU memory are given back (or no slots are left):
    /// their experts run on the CPU again, their pages read back into RAM ahead of that.
    fn free_blocks(&mut self, moe: &CpuMoe, bytes: usize) {
        // copies on their way into a slot land first
        let mut old = std::mem::replace(&mut self.uploader, ExpertUploader::new(self.slot_bytes));
        old.finish();
        while let Ok(d) = old.done.try_recv() {
            self.landed(d);
        }
        let base = moe.mmap.as_ptr() as usize;
        let mut freed = 0;
        while freed < bytes {
            let Some((buf, first)) = self.blocks.pop() else { break };
            for (l, e) in self.owner.drain(first..).flatten() {
                self.slot_of[l][e] = -1;
                os::prefetch(&moe.tensors[l].map(|t| (base + t.offset + e * t.stride, t.stride)));
            }
            self.addr.truncate(first);
            self.used.truncate(first);
            self.free.retain(|&s| s < first);
            freed += buf.bytes;
        }
    }

    /// GPU memory the slots take.
    fn bytes(&self) -> usize {
        self.blocks.iter().map(|(b, _)| b.bytes).sum()
    }

    /// Gate and up as GPU Q3_K (112-byte blocks: the 16-byte header first) into `out`, then the down's
    /// blocks as they are; `raw` (2-bit experts): all three as they are.
    fn pack(&self, moe: &CpuMoe, l: usize, e: usize, out: &mut [u8]) -> usize {
        let ex = moe.expert(l, e);
        Self::pack_parts(ex.gate.data, ex.up.data, ex.down.data, self.gate_bytes[l], iq2_fmt(&moe.tensors[l]).is_some(), out)
    }

    fn pack_parts(gate: &[u8], up: &[u8], down: &[u8], gate_bytes: usize, raw: bool, out: &mut [u8]) -> usize {
        if raw {
            out[..gate.len()].copy_from_slice(gate);
            out[gate_bytes..gate_bytes + up.len()].copy_from_slice(up);
            out[2 * gate_bytes..2 * gate_bytes + down.len()].copy_from_slice(down);
            return 2 * gate_bytes + down.len();
        }
        for (m, dst) in [(gate, 0), (up, gate_bytes)] {
            for (blk, o) in m.chunks_exact(110).zip(out[dst..dst + gate_bytes].chunks_exact_mut(Q3K_GPU_BYTES)) {
                q3k_repack(blk, o);
            }
        }
        out[2 * gate_bytes..2 * gate_bytes + down.len()].copy_from_slice(down);
        2 * gate_bytes + down.len()
    }

    /// An uncached expert into the next scratch slot (asynchronously, on the launch stream): returns
    /// its gate, up and down addresses and the scratch index to `release` after its kernels are queued.
    fn stage(&mut self, moe: &CpuMoe, l: usize, e: usize, stream: Ptr) -> Result<([u64; 3], usize), String> {
        let a = api()?;
        let i = self.next;
        self.next ^= 1;
        // the GPU is done with this pinned buffer and scratch slot (copy and kernels of two experts ago)
        a.check(unsafe { (a.f.cuEventSynchronize)(self.events[i]) }, "staging wait")?;
        let buf = unsafe { std::slice::from_raw_parts_mut(self.pinned[i], self.scratch_bytes) };
        let t = std::time::Instant::now();
        let bytes = self.pack(moe, l, e, buf);
        // a couple of MB from RAM take well under a millisecond; longer means the weights came off the disk
        let ms = t.elapsed().as_millis() as u64;
        if ms >= 3 {
            PROGRESS.waited_on_disk(ms);
        }
        let dst = self.scratch[i].ptr;
        a.check(unsafe { (a.f.cuMemcpyHtoDAsync_v2)(dst, self.pinned[i] as *const c_void, bytes, stream) }, "expert staging copy")?;
        self.staged += 1;
        let gb = self.gate_bytes[l] as u64;
        Ok(([dst, dst + gb, dst + 2 * gb], i))
    }

    fn release(&self, i: usize, stream: Ptr) -> Result<(), String> {
        let a = api()?;
        a.check(unsafe { (a.f.cuEventRecord)(self.events[i], stream) }, "event record")
    }

    fn slots(&self) -> usize {
        self.owner.len()
    }

    /// Device addresses of a cached expert's gate, up and down (a layer-`l` expert).
    fn ptrs(&self, slot: usize, l: usize) -> [u64; 3] {
        let (base, gb) = (self.addr[slot], self.gate_bytes[l] as u64);
        [base, base + gb, base + 2 * gb]
    }

    /// Marks the slots whose copies have landed as holding their experts.
    fn collect(&mut self) {
        while let Ok(d) = self.uploader.done.try_recv() {
            self.landed(d);
        }
    }

    fn landed(&mut self, (l, e, slot): (usize, usize, usize)) {
        self.slot_of[l][e] = slot as i32;
        self.queued[l][e] = false;
        self.used[slot] = self.clock;
        self.uploads += 1;
    }

    /// Layer l's picks: the cached ones count as used now; the others are queued for upload (a prompt batch
    /// only fills free slots; a token may also replace the least recently picked, up to its share).
    fn note(&mut self, moe: &CpuMoe, l: usize, picks: &[(usize, f32)], prompt: bool) {
        self.collect();
        self.clock += 1;
        if !self.cacheable[l] || self.slots() == 0 {
            return;
        }
        for &(e, _) in picks {
            let slot = self.slot_of[l][e];
            if slot >= 0 {
                self.used[slot as usize] = self.clock;
            }
        }
        for &(e, _) in picks {
            if self.slot_of[l][e] >= 0 || self.queued[l][e] {
                continue;
            }
            let base = moe.mmap.as_ptr() as usize;
            let ranges = |l: usize, e: usize| moe.tensors[l].map(|t| (base + t.offset + e * t.stride, t.stride));
            let mut evicted = None;
            let slot = match self.free.pop() {
                Some(s) => s,
                None if prompt || self.sent >= uploads_per_token() => continue,
                None => {
                    // the least recently picked (nothing picked this step: those may be in use on the GPU)
                    let (s, &when) = self.used.iter().enumerate().min_by_key(|(_, u)| **u).unwrap();
                    if when >= self.clock - 1 {
                        continue;
                    }
                    let (ol, oe) = self.owner[s].unwrap();
                    self.slot_of[ol][oe] = -1;
                    evicted = Some(ranges(ol, oe));
                    s
                }
            };
            self.sent += 1;
            moe_stat(MS_UPLOADS, 1);
            self.owner[slot] = Some((l, e));
            self.used[slot] = u64::MAX;
            self.queued[l][e] = true;
            let job = UploadJob { l, e, slot, src: ranges(l, e), gate_bytes: self.gate_bytes[l], raw: iq2_fmt(&moe.tensors[l]).is_some(), dst: self.addr[slot], evicted };
            if self.uploader.tx.as_ref().is_none_or(|tx| tx.send(job).is_err()) {
                // no upload thread: give the slot back
                self.owner[slot] = None;
                self.used[slot] = 0;
                self.queued[l][e] = false;
                self.free.push(slot);
                return;
            }
        }
    }

    /// Slots, how many hold an expert, and uploads so far (GHOST_INFER_LOAD_TIMES reports them).
    fn describe(&self) -> String {
        let used = self.slot_of.iter().flatten().filter(|s| **s >= 0).count();
        format!("{} slots ({} MB), {used} in use, {} uploads, {} staged for prompts", self.slots(), self.bytes() >> 20, self.uploads, self.staged)
    }

    /// Called once per forward pass.
    fn tick(&mut self, _n: usize) {
        self.sent = 0;
    }
}

/// The token embedding table: on the GPU (read by the embed kernel; needed there when the output is tied to it),
/// or, for a mixture of experts with an output of its own, left in the model file: a pass's rows are looked up on
/// the CPU and copied up, and its GPU memory (vocab x dim bytes) goes to the expert cache instead.
enum Embd {
    Gpu(GpuMatrix),
    Host { map: memmap2::Mmap, offset: usize, bytes: usize, ty: GgmlType, rows: usize, cols: usize },
    /// no table: the model is fed vectors (forward_vectors), as CSM's transformers are
    Vectors,
}

/// An extra output projection beside a model's own (CSM's per-codebook heads), applied to the final-normed
/// hidden state of the last pass (`GpuModel::head_logits`).
pub struct GpuHead(GpuMatrix);

/// The GPU side of CSM speech (csm.rs): a whole frame as one CUDA graph. From the backbone's last pass (its
/// logits and final-normed state) it samples the first code, runs the depth decoder on the projected state and
/// then on each code's row, with a head and a sampler per codebook, sums the frame's 32 rows into the
/// backbone's next input and runs the backbone on it. The host only reads the codes back. The two models
/// must share their kernels (GpuModel::share_kernels).
pub struct CsmFrames {
    proj: GpuHead,
    heads: Vec<GpuHead>,
    audio_embd: DeviceBuf, // backbone input rows (codebook, code), f16
    proj_embd: DeviceBuf,  // decoder input rows, already through its input projection, f16
    codes: DeviceBuf,      // the frame's codes (i32)
    params: DeviceBuf,     // [top-k, 1 / temperature] (f32)
    rng: DeviceBuf,        // [seed, draws so far] (u64)
    vocab: usize,
    codebooks: usize,
}

/// A captured frame, for one pair of sessions (it holds their addresses).
pub struct CsmGraph(Ptr);

impl Drop for CsmGraph {
    fn drop(&mut self) {
        if let Ok(a) = api() {
            unsafe { (a.f.cuGraphExecDestroy)(self.0) };
        }
    }
}

/// One attention block of Whisper: LayerNorm, q (with bias), k (without), v (with), and the output projection.
struct WhAttn {
    norm: (DeviceBuf, DeviceBuf),
    q: GpuMatrix,
    qb: DeviceBuf,
    k: GpuMatrix,
    v: GpuMatrix,
    vb: DeviceBuf,
    o: GpuMatrix,
    ob: DeviceBuf,
}

/// LayerNorm, then fc1, GELU, fc2 (each with a bias).
struct WhMlp {
    norm: (DeviceBuf, DeviceBuf),
    fc1: GpuMatrix,
    fc1b: DeviceBuf,
    fc2: GpuMatrix,
    fc2b: DeviceBuf,
}

/// Whisper speech-to-text on the GPU (whisper.rs). The encoder reads a clip's log-mel frames (two 3-tap
/// convolutions, then pre-norm transformer layers over half as many positions); the decoder reads tokens a
/// few at a time, attending to its own earlier ones and, through keys and values worked out once per clip, to
/// the encoder's output. Matrices run through the engine's matmuls (a bare GpuModel and a session for their
/// scratch); the rest is Whisper's own small kernels.
pub struct GpuWhisper {
    mm: GpuModel,
    s: GpuSession,
    dim: usize,
    heads: usize,
    ffn: usize,
    n_mels: usize,
    pub max_audio: usize, // encoder positions (1500: 30 s)
    pub max_text: usize,  // decoder positions (448)
    conv1: (GpuMatrix, DeviceBuf),
    conv2: (GpuMatrix, DeviceBuf),
    enc_pos: DeviceBuf,
    enc: Vec<(WhAttn, WhMlp)>,
    enc_norm: (DeviceBuf, DeviceBuf),
    dec: Vec<(WhAttn, WhAttn, WhMlp)>,
    dec_norm: (DeviceBuf, DeviceBuf),
    embd: GpuMatrix,
    // working buffers (rows of floats)
    inp: DeviceBuf,  // mel frames, then decoder input rows
    col: DeviceBuf,  // convolution rows
    x: DeviceBuf,    // the residual stream
    hn: DeviceBuf,   // normed rows
    q: DeviceBuf,
    k: DeviceBuf,
    v: DeviceBuf,
    att: DeviceBuf,
    big: DeviceBuf,  // conv1's output, the MLP's hidden rows
    cross: Vec<(DeviceBuf, DeviceBuf)>, // per decoder layer: the clip's keys and values
    selfkv: Vec<(DeviceBuf, DeviceBuf)>, // per decoder layer: its tokens' keys and values
    n_enc: usize,
    pub pos: usize, // decoder tokens read
}

impl GpuWhisper {
    pub fn load(g: &Gguf) -> Result<GpuWhisper, String> {
        let u = |k: &str| g.get(&format!("whisper.{k}")).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| format!("missing whisper.{k}"));
        let (dim, heads, ffn, n_mels, vocab) = (u("dim")?, u("heads")?, u("ffn")?, u("n_mels")?, u("vocab")?);
        let (max_audio, max_text) = (u("audio_positions")?, u("text_positions")?);
        if dim / heads != 64 {
            return Err(format!("Whisper heads of {} values are not supported (64)", dim / heads));
        }
        let cfg = Config {
            dim,
            n_layers: 0,
            n_heads: heads,
            n_kv_heads: heads,
            head_dim: 64,
            ffn_dim: ffn.max(3 * dim),
            vocab,
            max_ctx: 32,
            rope_dim: 64,
            rope_base: 10_000.0,
            rope_neox: false,
            rope_freqs: None,
            norm_eps: 1e-5,
            n_experts: 0,
            n_experts_used: 0,
            expert_ffn_dim: 0,
            mla: None,
            n_dense_lead: 0,
            n_shared_experts: 0,
            gating_sigmoid: false,
            expert_weights_norm: false,
            expert_weights_scale: 1.0,
            delta: None,
            attn_gated: false,
        };
        // its own non-blocking stream: it runs beside the language model, on a thread of its own
        let k = Arc::new(Kernels::compile_on(true)?);
        let mut up = Uploader::new()?;
        let up = &mut up;
        let one = |name: &str, up: &mut Uploader| upload_one(g, name, up);
        let vec = |name: &str, up: &mut Uploader| upload_vec(g, name, up);
        let norm = |name: &str, up: &mut Uploader| -> Result<(DeviceBuf, DeviceBuf), String> { Ok((vec(&format!("{name}.weight"), up)?, vec(&format!("{name}.bias"), up)?)) };
        let attn = |p: &str, up: &mut Uploader| -> Result<WhAttn, String> {
            Ok(WhAttn {
                norm: norm(&format!("{p}_norm"), up)?,
                q: one(&format!("{p}_q.weight"), up)?,
                qb: vec(&format!("{p}_q.bias"), up)?,
                k: one(&format!("{p}_k.weight"), up)?,
                v: one(&format!("{p}_v.weight"), up)?,
                vb: vec(&format!("{p}_v.bias"), up)?,
                o: one(&format!("{p}_o.weight"), up)?,
                ob: vec(&format!("{p}_o.bias"), up)?,
            })
        };
        let mlp = |p: &str, up: &mut Uploader| -> Result<WhMlp, String> {
            Ok(WhMlp { norm: norm(&format!("{p}ffn_norm"), up)?, fc1: one(&format!("{p}fc1.weight"), up)?, fc1b: vec(&format!("{p}fc1.bias"), up)?, fc2: one(&format!("{p}fc2.weight"), up)?, fc2b: vec(&format!("{p}fc2.bias"), up)? })
        };
        let enc = (0..u("encoder_layers")?).map(|i| Ok((attn(&format!("enc.blk.{i}.attn"), up)?, mlp(&format!("enc.blk.{i}."), up)?))).collect::<Result<Vec<_>, String>>()?;
        let dec = (0..u("decoder_layers")?)
            .map(|i| Ok((attn(&format!("dec.blk.{i}.attn"), up)?, attn(&format!("dec.blk.{i}.cross"), up)?, mlp(&format!("dec.blk.{i}."), up)?)))
            .collect::<Result<Vec<_>, String>>()?;
        // the cross-attention norms are named cross_norm, the self-attention ones attn_norm: as `attn` reads them
        let rows = 2 * max_audio; // mel frames
        let f = |n: usize| DeviceBuf::alloc(4 * n);
        let w = GpuWhisper {
            conv1: (one("enc.conv1.weight", up)?, vec("enc.conv1.bias", up)?),
            conv2: (one("enc.conv2.weight", up)?, vec("enc.conv2.bias", up)?),
            enc_pos: vec("enc.pos", up)?,
            enc_norm: norm("enc.norm", up)?,
            dec_norm: norm("dec.norm", up)?,
            embd: one("dec.embd", up)?,
            inp: f(rows * n_mels.max(dim))?,
            col: f(rows * 3 * n_mels.max(dim / 2).max(dim))?,
            x: f(max_audio * dim)?,
            hn: f(max_audio * dim)?,
            q: f(max_audio * dim)?,
            k: f(max_audio * dim)?,
            v: f(max_audio * dim)?,
            att: f(max_audio * dim)?,
            big: f((rows * dim).max(max_audio * ffn))?,
            cross: (0..dec.len()).map(|_| Ok((f(max_audio * dim)?, f(max_audio * dim)?))).collect::<Result<_, String>>()?,
            selfkv: (0..dec.len()).map(|_| Ok((f(max_text * dim)?, f(max_text * dim)?))).collect::<Result<_, String>>()?,
            s: GpuSession::new(&cfg, 32)?,
            mm: GpuModel {
                cfg,
                k,
                embd: Embd::Vectors,
                output: None,
                output_norm: DeviceBuf::alloc(4)?,
                rope_freqs: None,
                layers: Vec::new(),
                cache: Default::default(),
                moe: None,
                prefill_shared: None,
                device: String::new(),
                int8_q8: false,
                iq_scratch: None,
                _weights: None,
            },
            enc,
            dec,
            dim,
            heads,
            ffn,
            n_mels,
            max_audio,
            max_text,
            n_enc: 0,
            pos: 0,
        };
        up.finish()?;
        Ok(w)
    }

    /// `m` over `n` rows of `x` into rows of `out` (in chunks the matmul scratch holds).
    fn mm(&self, m: &GpuMatrix, x: u64, out: u64, n: usize, accumulate: bool) -> Result<(), String> {
        for c0 in (0..n).step_by(MAX_BATCH) {
            let c = (n - c0).min(MAX_BATCH);
            self.mm.matmul_one(m, x + (4 * c0 * m.cols) as u64, out + (4 * c0 * m.rows) as u64, c, accumulate, &self.s)?;
        }
        Ok(())
    }

    fn bias(&self, x: &DeviceBuf, b: &DeviceBuf, cols: usize, n: usize, gelu: bool) -> Result<(), String> {
        let k = &self.mm.k;
        k.launch(k.wh_bias, ((n * cols).div_ceil(256) as u32, 1), 256, 0, &[Arg::P(x.ptr), Arg::P(b.ptr), Arg::I(cols as i32), Arg::I(n as i32), Arg::I(gelu as i32)])
    }

    fn layernorm(&self, x: u64, (w, b): &(DeviceBuf, DeviceBuf), out: u64, n: usize) -> Result<(), String> {
        let k = &self.mm.k;
        k.launch(k.wh_layernorm, (n as u32, 1), 256, 0, &[Arg::P(x), Arg::P(w.ptr), Arg::P(b.ptr), Arg::P(out), Arg::I(self.dim as i32), Arg::F(1e-5)])
    }

    fn attention(&self, q: u64, keys: u64, values: u64, nq: usize, nk: usize, causal: i32) -> Result<(), String> {
        let k = &self.mm.k;
        let d = self.dim as i32;
        k.launch(k.wh_attention, (nq as u32, self.heads as u32), 128, (4 * nk) as u32, &[Arg::P(q), Arg::P(keys), Arg::P(values), Arg::P(self.att.ptr), Arg::I(nk as i32), Arg::I(d), Arg::I(d), Arg::I(self.heads as i32), Arg::I(causal), Arg::F(0.125)])
    }

    /// x += the attention block on `n` rows; `kv`: the keys and values to read (None: from these rows) and how
    /// many; `store`: where this block's own keys and values go (the decoder's cache, at row `at`).
    fn attn_block(&self, a: &WhAttn, n: usize, kv: Option<(u64, u64, usize)>, store: Option<(u64, u64, usize)>, causal: i32) -> Result<(), String> {
        let d = self.dim;
        self.layernorm(self.x.ptr, &a.norm, self.hn.ptr, n)?;
        self.mm(&a.q, self.hn.ptr, self.q.ptr, n, false)?;
        self.bias(&self.q, &a.qb, d, n, false)?;
        let (keys, values, nk) = match (kv, store) {
            (Some(kv), _) => kv,
            (None, Some((kc, vc, at))) => {
                let (kr, vr) = (kc + (4 * at * d) as u64, vc + (4 * at * d) as u64);
                self.mm(&a.k, self.hn.ptr, kr, n, false)?;
                self.mm(&a.v, self.hn.ptr, vr, n, false)?;
                let k = &self.mm.k;
                k.launch(k.wh_bias, ((n * d).div_ceil(256) as u32, 1), 256, 0, &[Arg::P(vr), Arg::P(a.vb.ptr), Arg::I(d as i32), Arg::I(n as i32), Arg::I(0)])?;
                (kc, vc, at + n)
            }
            (None, None) => {
                self.mm(&a.k, self.hn.ptr, self.k.ptr, n, false)?;
                self.mm(&a.v, self.hn.ptr, self.v.ptr, n, false)?;
                self.bias(&self.v, &a.vb, d, n, false)?;
                (self.k.ptr, self.v.ptr, n)
            }
        };
        self.attention(self.q.ptr, keys, values, n, nk, causal)?;
        self.mm(&a.o, self.att.ptr, self.x.ptr, n, true)?;
        self.bias(&self.x, &a.ob, d, n, false)
    }

    fn mlp_block(&self, m: &WhMlp, n: usize) -> Result<(), String> {
        self.layernorm(self.x.ptr, &m.norm, self.hn.ptr, n)?;
        self.mm(&m.fc1, self.hn.ptr, self.big.ptr, n, false)?;
        self.bias(&self.big, &m.fc1b, self.ffn, n, true)?;
        self.mm(&m.fc2, self.big.ptr, self.x.ptr, n, true)?;
        self.bias(&self.x, &m.fc2b, self.dim, n, false)
    }

    /// Encodes a clip: `mel` is rows of `n_mels` (frames, 100 a second; at most 2 * max_audio). Leaves the
    /// cross-attention keys and values for the decoder, and starts it afresh.
    pub fn encode(&mut self, mel: &[f32]) -> Result<(), String> {
        let (d, k) = (self.dim, self.mm.k.clone());
        let t = (mel.len() / self.n_mels).min(2 * self.max_audio);
        let n = t.div_ceil(2);
        if n == 0 {
            return Err("no audio".into());
        }
        self.mm.k.h2d(self.inp.ptr, &mel[..t * self.n_mels])?;
        let im2col = |x: u64, out: u64, nin: usize, nout: usize, c: usize, stride: usize| {
            k.launch(k.wh_im2col, ((nout * 3 * c).div_ceil(256) as u32, 1), 256, 0, &[Arg::P(x), Arg::P(out), Arg::I(nin as i32), Arg::I(nout as i32), Arg::I(c as i32), Arg::I(stride as i32)])
        };
        im2col(self.inp.ptr, self.col.ptr, t, t, self.n_mels, 1)?;
        self.mm(&self.conv1.0, self.col.ptr, self.big.ptr, t, false)?;
        self.bias(&self.big, &self.conv1.1, d, t, true)?;
        im2col(self.big.ptr, self.col.ptr, t, n, d, 2)?;
        self.mm(&self.conv2.0, self.col.ptr, self.x.ptr, n, false)?;
        self.bias(&self.x, &self.conv2.1, d, n, true)?;
        k.launch(k.wh_add, ((n * d).div_ceil(256) as u32, 1), 256, 0, &[Arg::P(self.x.ptr), Arg::P(self.enc_pos.ptr), Arg::I((n * d) as i32)])?;
        for (a, m) in &self.enc {
            self.attn_block(a, n, None, None, -1)?;
            self.mlp_block(m, n)?;
        }
        self.layernorm(self.x.ptr, &self.enc_norm, self.hn.ptr, n)?;
        for ((_, cross, _), (kc, vc)) in self.dec.iter().zip(&self.cross) {
            self.mm(&cross.k, self.hn.ptr, kc.ptr, n, false)?;
            self.mm(&cross.v, self.hn.ptr, vc.ptr, n, false)?;
            self.bias(vc, &cross.vb, d, n, false)?;
        }
        self.n_enc = n;
        self.pos = 0;
        Ok(())
    }

    /// Reads decoder input rows (token embedding + position, `dim` floats each) at positions `pos..`; returns the
    /// last one's logits.
    pub fn decode(&mut self, rows: &[f32], logits: &mut Vec<f32>) -> Result<(), String> {
        let d = self.dim;
        let n = rows.len() / d;
        if self.pos + n > self.max_text {
            return Err("the transcript is longer than Whisper's 448 tokens".into());
        }
        self.mm.k.h2d(self.x.ptr, rows)?;
        for ((own, cross, m), ((kc, vc), (ck, cv))) in self.dec.iter().zip(self.selfkv.iter().zip(&self.cross)) {
            self.attn_block(own, n, None, Some((kc.ptr, vc.ptr, self.pos)), self.pos as i32)?;
            self.attn_block(cross, n, Some((ck.ptr, cv.ptr, self.n_enc)), None, -1)?;
            self.mlp_block(m, n)?;
        }
        self.pos += n;
        let last = self.x.ptr + (4 * (n - 1) * d) as u64;
        self.layernorm(last, &self.dec_norm, self.hn.ptr, 1)?;
        self.mm(&self.embd, self.hn.ptr, self.big.ptr, 1, false)?;
        logits.resize(self.embd.rows, 0.0);
        self.mm.k.d2h(logits, self.big.ptr)
    }
}

/// Makes the GPU context current on this thread (a thread other than the one that loaded a model).
pub fn bind_thread() -> Result<(), String> {
    let a = api()?;
    a.check(unsafe { (a.f.cuCtxSetCurrent)(a.ctx as Ptr) }, "bind the GPU context")
}

impl CsmFrames {
    pub fn load(bg: &Gguf, dg: &Gguf, codebooks: usize, vocab: usize) -> Result<CsmFrames, String> {
        let mut up = Uploader::new()?;
        let f16 = |g: &Gguf, name: &str, up: &mut Uploader| -> Result<DeviceBuf, String> {
            let t = g.tensor(name).ok_or_else(|| format!("missing tensor {name}"))?;
            if t.ty != GgmlType::F16 {
                return Err(format!("{name} is {}, not F16", t.ty.name()));
            }
            let buf = DeviceBuf::alloc(t.n_bytes)?;
            up.put(buf.ptr, &g.mmap[t.offset..t.offset + t.n_bytes])?;
            Ok(buf)
        };
        let frames = CsmFrames {
            proj: GpuHead(upload_one(dg, "csm.proj.weight", &mut up)?),
            heads: (1..codebooks).map(|i| upload_one(dg, &format!("csm.head.{i}.weight"), &mut up).map(GpuHead)).collect::<Result<_, _>>()?,
            audio_embd: f16(bg, "csm.audio_embd.weight", &mut up)?,
            proj_embd: f16(dg, "csm.proj_embd.weight", &mut up)?,
            codes: DeviceBuf::alloc(4 * codebooks)?,
            params: DeviceBuf::alloc(8)?,
            rng: DeviceBuf::alloc(16)?,
            vocab,
            codebooks,
        };
        up.finish()?;
        if frames.heads.iter().any(|h| h.0.rows != vocab) {
            return Err("a codebook head does not have one row per code".into());
        }
        Ok(frames)
    }

    /// The decoder's input projection, as a head of the backbone.
    pub fn proj(&self) -> &GpuHead {
        &self.proj
    }

    /// The head of codebook `cb` (1 ..).
    pub fn head(&self, cb: usize) -> &GpuHead {
        &self.heads[cb - 1]
    }

    /// Sampling for the next frames: temperature (0: greedy), top-k, and a seed.
    pub fn set_sampling(&self, backbone: &GpuModel, temperature: f32, top_k: usize, seed: u64) -> Result<(), String> {
        backbone.k.h2d(self.params.ptr, &[top_k as f32, if temperature > 0.0 { 1.0 / temperature } else { 0.0 }])?;
        backbone.k.h2d(self.rng.ptr, &[seed, 0u64])
    }

    /// The next frame's codes, after a backbone pass that left its logits on the GPU (forward_vectors with
    /// logits); the backbone has run on the frame when this returns, ready for the one after.
    pub fn step(&self, backbone: &GpuModel, decoder: &GpuModel, bb: &mut GpuSession, dec: &mut GpuSession, graph: &mut Option<CsmGraph>) -> Result<Vec<u32>, String> {
        if !Arc::ptr_eq(&backbone.k, &decoder.k) {
            return Err("CSM's two models must share their kernels".into());
        }
        if bb.pos >= bb.ctx {
            return Err(format!("context full ({} frames)", bb.ctx));
        }
        let (a, k) = (api()?, &backbone.k);
        k.h2d(bb.pos_dev.ptr, &[bb.pos as i32])?;
        if graph.is_none() {
            const RELAXED: i32 = 2; // CU_STREAM_CAPTURE_MODE_RELAXED
            a.check(unsafe { (a.f.cuStreamBeginCapture_v2)(k.stream, RELAXED) }, "begin graph capture")?;
            let launched = self.launches(backbone, decoder, bb, dec);
            let mut g: Ptr = std::ptr::null_mut();
            let ended = unsafe { (a.f.cuStreamEndCapture)(k.stream, &mut g) };
            dec.pos = 0;
            launched?;
            a.check(ended, "end graph capture")?;
            let mut exec: Ptr = std::ptr::null_mut();
            let made = unsafe { (a.f.cuGraphInstantiateWithFlags)(&mut exec, g, 0) };
            unsafe { (a.f.cuGraphDestroy)(g) };
            a.check(made, "instantiate graph")?;
            *graph = Some(CsmGraph(exec));
        }
        a.check(unsafe { (a.f.cuGraphLaunch)(graph.as_ref().unwrap().0, k.stream) }, "graph launch")?;
        let mut codes = vec![0i32; self.codebooks];
        k.d2h(&mut codes, self.codes.ptr)?;
        bb.pos += 1;
        Ok(codes.into_iter().map(|c| c as u32).collect())
    }

    /// The frame's kernels (see the type); the decoder's positions (0 ..) are the same every frame, so they go
    /// into the graph as they are, while the backbone reads its position from `bb.pos_dev`.
    fn launches(&self, backbone: &GpuModel, decoder: &GpuModel, bb: &mut GpuSession, dec: &mut GpuSession) -> Result<(), String> {
        let k = &backbone.k;
        let (v, dd, cb) = (self.vocab, decoder.cfg.dim, self.codebooks);
        let sample = |logits: u64, slot: usize| {
            k.launch(k.csm_sample, (1, 1), 1024, 0, &[Arg::P(logits), Arg::I(v as i32), Arg::P(self.params.ptr), Arg::P(self.rng.ptr), Arg::P(self.codes.ptr), Arg::I(slot as i32)])
        };
        let embed = |table: &DeviceBuf, cols: usize, first: usize, m: usize, out: u64| {
            k.launch(k.csm_embed, (cols.div_ceil(256) as u32, 1), 256, 0, &[Arg::P(table.ptr), Arg::I(cols as i32), Arg::P(self.codes.ptr), Arg::I(first as i32), Arg::I(m as i32), Arg::I(v as i32), Arg::P(out)])
        };
        bb.exact.set(false);
        dec.exact.set(false);
        sample(bb.logits_dev.ptr, 0)?;
        backbone.matmul_one(&self.proj.0, bb.xb.ptr, dec.x.ptr, 1, false, bb)?;
        embed(&self.proj_embd, dd, 0, 1, dec.x.at(dd * 4))?;
        dec.pos = 0;
        decoder.launches(2, dec, Out::Hidden, 0, false)?;
        dec.pos = 2;
        for i in 1..cb {
            decoder.matmul_one(&self.heads[i - 1].0, dec.xb.ptr, dec.logits_dev.ptr, 1, false, dec)?;
            sample(dec.logits_dev.ptr, i)?;
            if i + 1 < cb {
                embed(&self.proj_embd, dd, i, 1, dec.x.ptr)?;
                decoder.launches(1, dec, Out::Hidden, 0, false)?;
                dec.pos += 1;
            }
        }
        // the frame as the backbone's next input, and its pass
        embed(&self.audio_embd, backbone.cfg.dim, 0, cb, bb.x.ptr)?;
        backbone.launches(1, bb, Out::Last, bb.pos_dev.ptr, true)
    }
}

pub struct GpuModel {
    pub cfg: Config,
    k: Arc<Kernels>,
    embd: Embd,
    output: Option<GpuMatrix>, // None = tied to the embedding (then on the GPU), or none (Embd::Vectors)
    output_norm: DeviceBuf,
    rope_freqs: Option<DeviceBuf>,
    layers: Vec<GpuLayer>,
    /// the GPU-resident experts, made on the first forward pass from the memory left then (before `moe`:
    /// its upload thread reads the mapping, so it stops first)
    cache: std::cell::RefCell<Option<ExpertCache>>,
    moe: Option<CpuMoe>,
    prefill_shared: Option<usize>,
    pub device: String,
    /// Q8_0 weights decode with int8 activations too (for draft models; otherwise exact f32)
    pub int8_q8: bool,
    /// 2-bit weights (Fmt::Iq2) as Q8 for prompt batches: int8 values, f32 scales
    iq_scratch: Option<(DeviceBuf, DeviceBuf)>,
    /// the weights' memory (dropped after the buffers pointing into it, which free nothing)
    _weights: Option<Arena>,
}

impl GpuModel {
    pub fn load(g: &Gguf, cfg: Config) -> Result<GpuModel, String> {
        Self::load_with(g, cfg, false)
    }

    /// load, with private: its kernels on a non-blocking stream of their own (see Kernels::private), for a
    /// model that runs on another thread beside the language model.
    pub fn load_with(g: &Gguf, cfg: Config, private: bool) -> Result<GpuModel, String> {
        if let Some(m) = cfg.mla {
            if m.kv_lora != 512 || cfg.n_heads > 32 || (m.kv_lora + cfg.rope_dim) % 32 != 0 || m.kv_lora + cfg.rope_dim > 32 * 24 {
                return Err(format!("GPU backend: latent attention with {} heads and a {}-wide cache row is not supported (512 + RoPE, up to 32 heads)", cfg.n_heads, m.kv_lora + cfg.rope_dim));
            }
        } else if cfg.attn_gated {
            // wide heads go to attention_wide (a multiple of 128, up to 256); DeltaNet heads to delta_scan
            if cfg.head_dim % 128 != 0 || cfg.head_dim > 256 || cfg.n_heads % cfg.n_kv_heads != 0 {
                return Err(format!("GPU backend: gated attention with head size {} is not supported (128 or 256)", cfg.head_dim));
            }
            if cfg.delta.is_some_and(|d| d.head != 128 || d.conv > 8) {
                return Err("GPU backend: Gated DeltaNet heads must be 128 wide".into());
            }
        } else if cfg.head_dim > ATT_MAX_HD || ATT_MAX_HD % cfg.head_dim != 0 || cfg.head_dim % 4 != 0 {
            return Err(format!("GPU backend: head size {} not supported (needs a divisor of {ATT_MAX_HD})", cfg.head_dim));
        }
        let t0 = std::time::Instant::now();
        let a = api()?;
        let t_api = t0.elapsed();
        let k = Arc::new(Kernels::compile_on(private)?);
        let t_kernels = t0.elapsed();
        let mut up = Uploader::new()?;
        up.arena = Arena::new().ok();
        let up = &mut up;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}.weight");
            let attn_norm = upload_vec(g, &p("attn_norm"), up)?;
            let delta = if cfg.recurrent(i) {
                let v = |s: &str| format!("blk.{i}.{s}");
                // [q k v | z | ba]: beta and alpha from one tensor or (tiled) two, side by side either way
                let mut parts = vec![p("attn_qkv"), p("attn_gate")];
                if cfg.delta.is_some_and(|d| d.tiled) {
                    parts.extend([p("ssm_beta"), p("ssm_alpha")]);
                } else {
                    parts.push(p("ssm_ba"));
                }
                Some(GpuDelta {
                    proj: upload_parts(g, &parts, up)?,
                    conv: upload_vec(g, &v("ssm_conv1d.weight"), up)?,
                    dt_bias: upload_vec(g, &v("ssm_dt.bias"), up)?,
                    a: upload_vec(g, &v("ssm_a"), up)?,
                    norm: upload_vec(g, &v("ssm_norm.weight"), up)?,
                })
            } else {
                None
            };
            let (qkv, mla) = match cfg.mla {
                _ if delta.is_some() => (Vec::new(), None),
                Some(m) => {
                    let q_a = if m.q_lora > 0 { Some((upload_one(g, &p("attn_q_a"), up)?, upload_vec(g, &p("attn_q_a_norm"), up)?)) } else { None };
                    let mla = GpuMla {
                        q: upload_one(g, &p(if m.q_lora > 0 { "attn_q_b" } else { "attn_q" }), up)?,
                        q_a,
                        kv_a: upload_one(g, &p("attn_kv_a_mqa"), up)?,
                        kv_norm: upload_vec(g, &p("attn_kv_a_norm"), up)?,
                        k_b: upload_f16(g, &p("attn_k_b"), up)?,
                        v_b: upload_f16(g, &p("attn_v_b"), up)?,
                    };
                    (Vec::new(), Some(mla))
                }
                None => (upload_parts(g, &[p("attn_q"), p("attn_k"), p("attn_v")], up)?, None),
            };
            let q_norm = if g.tensor(&p("attn_q_norm")).is_some() { Some(upload_vec(g, &p("attn_q_norm"), up)?) } else { None };
            let k_norm = if g.tensor(&p("attn_k_norm")).is_some() { Some(upload_vec(g, &p("attn_k_norm"), up)?) } else { None };
            let wo = upload_one(g, &p(if delta.is_some() { "ssm_out" } else { "attn_output" }), up)?;
            // qwen3next names the norm before the feed-forward after the attention it follows
            let ffn_norm = upload_vec(g, &p(if g.tensor(&p("ffn_norm")).is_some() { "ffn_norm" } else { "post_attention_norm" }), up)?;
            let ffn = if cfg.is_moe() && i >= cfg.n_dense_lead {
                let shared = if g.tensor(&p("ffn_gate_shexp")).is_some() {
                    Some((upload_parts(g, &[p("ffn_gate_shexp"), p("ffn_up_shexp")], up)?, upload_one(g, &p("ffn_down_shexp"), up)?))
                } else {
                    None
                };
                let gate = if g.tensor(&p("ffn_gate_inp_shexp")).is_some() { Some(upload_f32(g, &p("ffn_gate_inp_shexp"), up)?) } else { None };
                GpuFfn::Moe { shared, gate }
            } else {
                GpuFfn::Dense { gate_up: upload_parts(g, &[p("ffn_gate"), p("ffn_up")], up)?, down: upload_one(g, &p("ffn_down"), up)? }
            };
            layers.push(GpuLayer { attn_norm, qkv, q_norm, k_norm, mla, delta, wo, ffn_norm, ffn });
        }
        let moe = if cfg.is_moe() { Some(CpuMoe::load(g, &cfg)?) } else { None };
        let output = if g.tensor("output.weight").is_some() { Some(upload_one(g, "output.weight", up)?) } else { None };
        // in RAM, a row read per token: a mixture of experts' (its GPU memory goes to the expert cache), a model
        // whose loader asks (ghost.embd_on_host, as maya1.rs does: it only ever reads its prompt's rows), or a
        // big table (Mistral Nemo's 131k rows are 360 MB at Q4_K: on an 8 GB card, 2300 more tokens of context)
        let big = g.tensor("token_embd.weight").is_some_and(|t| t.n_bytes >= 256 << 20);
        let host = cfg.is_moe() || big || matches!(g.get("ghost.embd_on_host"), Some(crate::gguf::Value::Bool(true)));
        let embd = match g.tensor("token_embd.weight") {
            Some(t) if host && output.is_some() && t.dims.len() == 2 && t.ty.block().is_some() => {
                let file = std::fs::File::open(&g.path).map_err(|e| format!("cannot open {}: {e}", g.path.display()))?;
                let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("cannot map {}: {e}", g.path.display()))?;
                Embd::Host { map, offset: t.offset, bytes: t.n_bytes, ty: t.ty, rows: t.dims[1] as usize, cols: t.dims[0] as usize }
            }
            Some(_) => Embd::Gpu(upload_parts_as(g, &["token_embd.weight".to_string()], up, false)?.pop().unwrap().0),
            None => Embd::Vectors,
        };
        let rope_freqs = match &cfg.rope_freqs {
            Some(f) => Some(up.put_vec(f)?),
            None => None,
        };
        // the conversion scratch of 2-bit weights (the most any one matmul converts) and of 3-bit ones (a matrix,
        // Q3K_CONV values at most)
        let one = |m: &GpuMatrix| match m.fmt {
            Fmt::Iq2 => m.rows * m.cols,
            Fmt::Q3K => (m.rows * m.cols).min(Q3K_CONV),
            _ => 0,
        };
        let iq2 = |parts: &[(GpuMatrix, usize)]| {
            let all = parts.iter().filter(|(m, _)| m.fmt == Fmt::Iq2).map(|(m, _)| m.rows * m.cols).sum::<usize>();
            parts.iter().map(|(m, _)| one(m)).fold(all, usize::max)
        };
        let most = layers
            .iter()
            .map(|l| {
                let ffn = match &l.ffn {
                    GpuFfn::Dense { gate_up, down } => iq2(gate_up).max(one(down)),
                    GpuFfn::Moe { shared: Some((gate_up, down)), .. } => iq2(gate_up).max(one(down)),
                    GpuFfn::Moe { shared: None, .. } => 0,
                };
                iq2(&l.qkv).max(l.delta.as_ref().map_or(0, |d| iq2(&d.proj))).max(one(&l.wo)).max(ffn)
            })
            .chain(output.as_ref().map(one))
            .max()
            .unwrap_or(0);
        let iq_scratch = if most > 0 { Some((DeviceBuf::alloc(most)?, DeviceBuf::alloc(most / 32 * 4)?)) } else { None };

        // prompt attention: dynamic shared memory above the default 48 KB has to be requested per kernel
        let ld = cfg.head_dim + 1;
        let bytes = ((2 * PREFILL_KT + PREFILL_QT) * ld + PREFILL_QT * (PREFILL_KT + 1)) * 4;
        const MAX_DYNAMIC_SHARED: i32 = 8; // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        let prefill_shared = (bytes <= 48 * 1024 || unsafe { (a.f.cuFuncSetAttribute)(k.attention_prefill, MAX_DYNAMIC_SHARED, bytes as i32) } == 0).then_some(bytes);
        if let Some(m) = cfg.mla {
            let h = cfg.n_heads;
            let bytes = (h * (m.kv_lora + cfg.rope_dim) + h * MLA_T + 3 * h) * 4;
            if bytes > 48 * 1024 && unsafe { (a.f.cuFuncSetAttribute)(k.mla_attention, MAX_DYNAMIC_SHARED, bytes as i32) } != 0 {
                return Err(format!("GPU backend: latent attention needs {bytes} bytes of shared memory per block"));
            }
        }

        let device = format!("{} (sm_{}{}, NVRTC {})", a.device_name, a.compute.0, a.compute.1, std::path::Path::new(&a.nvrtc_path).file_name().and_then(|n| n.to_str()).unwrap_or("?"));
        // the logits buffer is sized from cfg.vocab; a mismatch would write past it
        let out_rows = match (&output, &embd) {
            (Some(o), _) => Some(o.rows),
            (None, Embd::Gpu(e)) => Some(e.rows),
            (None, Embd::Host { .. }) => unreachable!("a tied output keeps the embedding on the GPU"),
            (None, Embd::Vectors) => None, // hidden states only
        };
        if let Some(rows) = out_rows.filter(|&r| r != cfg.vocab) {
            return Err(format!("output layer has {rows} rows but the vocabulary has {} tokens", cfg.vocab));
        }
        let output_norm = upload_vec(g, "output_norm.weight", up)?;
        up.finish()?;
        let model = GpuModel { output_norm, rope_freqs, cfg, k, embd, output, layers, moe, cache: Default::default(), prefill_shared, device, int8_q8: false, iq_scratch, _weights: up.arena.take() };
        if std::env::var_os("GHOST_INFER_LOAD_TIMES").is_some() {
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
            eprintln!(
                "[GPU load: driver {:.0} ms, kernels {:.0} ms, weights {:.0} ms]",
                ms(t_api),
                ms(t_kernels - t_api),
                ms(t0.elapsed() - t_kernels)
            );
        }
        Ok(model)
    }

    /// The output projection: its own, or the embedding it is tied to (which then stays on the GPU).
    fn output_matrix(&self) -> &GpuMatrix {
        match (&self.output, &self.embd) {
            (Some(o), _) => o,
            (None, Embd::Gpu(e)) => e,
            (None, Embd::Host { .. }) => unreachable!("a tied output keeps the embedding on the GPU"),
            (None, Embd::Vectors) => unreachable!("a model without an output gives hidden states only"),
        }
    }

    /// With GHOST_INFER_PROFILE set, waits for the GPU after `f` so its time lands in bucket `op`.
    fn timed(&self, op: usize, f: impl FnOnce() -> Result<(), String>) -> Result<(), String> {
        if !profile::enabled() {
            return f();
        }
        profile::time(op, || {
            f()?;
            let a = api()?;
            a.check(unsafe { (a.f.cuCtxSynchronize)() }, "synchronize")
        })
    }

    /// For each part, `out[t*out_stride + off + r] (+)= W[r] · x[t]` over `n` input rows of `cols` floats.
    fn matmul(&self, parts: &[(&GpuMatrix, usize)], x: u64, out: u64, out_stride: usize, n: usize, accumulate: bool, s: &GpuSession) -> Result<(), String> {
        let k = &self.k;
        let cols = parts[0].0.cols;
        // 2-bit weights outside decoding (prompt batches, exact mode): converted to Q8 in the scratch first
        if parts.iter().any(|(m, _)| m.fmt == Fmt::Iq2) && !self.dp4a_path(parts, n, s) {
            let (sq, sd) = self.iq_scratch.as_ref().ok_or("no scratch for 2-bit weights")?;
            let (mut qo, mut views) = (0usize, Vec::new());
            for &(m, _) in parts.iter().filter(|(m, _)| m.fmt == Fmt::Iq2) {
                let blocks = m.rows * m.cols / 32;
                let (q, d) = (sq.ptr + qo as u64, sd.ptr + (qo / 32 * 4) as u64);
                k.launch(k.iq2_to_q8, (blocks.div_ceil(8) as u32, 1), 256, 0, &[Arg::P(m.q.ptr), Arg::P(q), Arg::P(d), Arg::I(blocks as i32), Arg::I(m.iq)])?;
                views.push(GpuMatrix::view(Fmt::Q8, q, Some(d), m.rows, m.cols));
                qo += blocks * 32;
            }
            let mut vs = views.iter();
            let conv: Vec<(&GpuMatrix, usize)> = parts.iter().map(|&(m, off)| if m.fmt == Fmt::Iq2 { (&**vs.next().unwrap(), off) } else { (m, off) }).collect();
            return self.matmul(&conv, x, out, out_stride, n, accumulate, s);
        }
        // 3-bit weights likewise, a slice of rows at a time through the scratch (a 12B's [gate | up] whole would be
        // 150 MB); the launches queue in order, so a slice is converted only after the last one's matmul read it
        if parts.iter().any(|(m, _)| m.fmt == Fmt::Q3K) && !self.dp4a_path(parts, n, s) {
            let (sq, sd) = self.iq_scratch.as_ref().ok_or("no scratch for 3-bit weights")?;
            for &(m, off) in parts {
                if m.fmt != Fmt::Q3K {
                    self.matmul(&[(m, off)], x, out, out_stride, n, accumulate, s)?;
                    continue;
                }
                let slice = (sq.bytes / m.cols / TILE * TILE).max(TILE);
                for r0 in (0..m.rows).step_by(slice) {
                    let rows = slice.min(m.rows - r0);
                    let blocks = rows * m.cols / 32;
                    let w = m.q.ptr + (r0 * (m.cols / 256) * Q3K_GPU_BYTES) as u64;
                    k.launch(k.q3k_to_q8, (blocks.div_ceil(8) as u32, 1), 256, 0, &[Arg::P(w), Arg::P(sq.ptr), Arg::P(sd.ptr), Arg::I(blocks as i32)])?;
                    let v = GpuMatrix::view(Fmt::Q8, sq.ptr, Some(sd.ptr), rows, m.cols);
                    self.matmul(&[(&*v, off + r0)], x, out, out_stride, n, accumulate, s)?;
                }
            }
            return Ok(());
        }
        if n >= TILED_MIN_BATCH && !self.dp4a_path(parts, n, s) {
            if let Some(mk) = k.matmul_mma_k.filter(|_| parts.iter().all(|(m, _)| m.fmt != Fmt::Q8)) {
                // K-quant prompts: one activation scale per 128 (or 256) values keeps the tensor-core epilogue integer
                let chunks = n * cols / qch();
                k.launch(k.quantize_q8k, ((chunks * 32).div_ceil(256) as u32, 1), 256, 0, &[Arg::P(x), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.xsum.ptr), Arg::I(chunks as i32)])?;
                for &(m, off) in parts {
                    k.launch(
                        mk[(m.fmt == Fmt::Q6K) as usize],
                        (m.rows.div_ceil(TILE) as u32, n.div_ceil(TILE) as u32),
                        128,
                        0,
                        &[Arg::P(m.q.ptr), Arg::P(m.d_ptr()), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.xsum.ptr), Arg::P(out), Arg::I(m.rows as i32), Arg::I(m.cols as i32), Arg::I(n as i32), Arg::I(accumulate as i32), Arg::I(out_stride as i32), Arg::I(off as i32)],
                    )?;
                }
                return Ok(());
            }
            // prompts: quantise the activations once, then int8 tiles with dp4a
            let blocks = n * cols / 32;
            k.launch(k.quantize_q8, ((blocks * 32).div_ceil(256) as u32, 1), 256, 0, &[Arg::P(x), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.xsum.ptr), Arg::I(blocks as i32)])?;
            for &(m, off) in parts {
                let (func, threads) = match k.matmul_mma {
                    Some(f) => (f, 128),
                    None => (k.matmul_tiled, 256),
                };
                k.launch(
                    func,
                    (m.rows.div_ceil(TILE) as u32, n.div_ceil(TILE) as u32),
                    threads,
                    0,
                    &[Arg::I(m.fmt as i32), Arg::P(m.q.ptr), Arg::P(m.d_ptr()), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.xsum.ptr), Arg::P(out), Arg::I(m.rows as i32), Arg::I(m.cols as i32), Arg::I(n as i32), Arg::I(accumulate as i32), Arg::I(out_stride as i32), Arg::I(off as i32)],
                )?;
            }
            return Ok(());
        }
        // decoding (up to DECODE_MAX_BATCH tokens, or a verify batch): int8 activations + dp4a or tensor cores
        // (GHOST_INFER_EXACT=1 keeps f32)
        if self.dp4a_path(parts, n, s) {
            let blocks = n * cols / 32;
            k.launch(k.quantize_q8, ((blocks * 32).div_ceil(256) as u32, 1), 256, 0, &[Arg::P(x), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.xsum.ptr), Arg::I(blocks as i32)])?;
            return self.matmul_dp4a(parts, out, out_stride, n, accumulate, s);
        }
        // otherwise f32 activations; 256 threads = 8 warps, one row each for Q8_0, RPW rows each for K-quants
        for &(m, off) in parts {
            let tail = [Arg::P(x), Arg::P(out), Arg::I(m.rows as i32), Arg::I(m.cols as i32), Arg::I(n as i32), Arg::I(accumulate as i32), Arg::I(out_stride as i32), Arg::I(off as i32)];
            let (variant, tb) = if n == 1 { (0, 1) } else { (1, KQ_TB) };
            let kq_grid = (m.rows.div_ceil(8 * kq_rpw()) as u32, n.div_ceil(tb) as u32);
            match m.fmt {
                Fmt::Q8 => k.launch(k.matmul_q8, (m.rows.div_ceil(8) as u32, n.div_ceil(MATMUL_TB) as u32), 256, 0, &[&[Arg::P(m.q.ptr), Arg::P(m.d_ptr())][..], &tail].concat())?,
                Fmt::Q4K => k.launch(k.matmul_q4k[variant], kq_grid, 256, 0, &[&[Arg::P(m.q.ptr), Arg::P(0)][..], &tail].concat())?,
                Fmt::Q6K => k.launch(k.matmul_q6k[variant], kq_grid, 256, 0, &[&[Arg::P(m.q.ptr), Arg::P(m.d_ptr())][..], &tail].concat())?,
                Fmt::Iq2 | Fmt::Q3K => return Err("2- or 3-bit weights reached the f32 matmul unconverted".into()),
            }
        }
        Ok(())
    }

    /// Decoding batches (1..=DECODE_MAX_BATCH tokens) with K-quant weights go through int8 activations and
    /// dp4a; Q8_0 too when the model is a draft (`int8_q8`), keeping exact f32 decoding otherwise. So do
    /// verify batches up to VERIFY_MAX_BATCH (`s.exact`) when the tensor-core verify kernels take them.
    fn dp4a_path(&self, parts: &[(&GpuMatrix, usize)], n: usize, s: &GpuSession) -> bool {
        if crate::quant::exact_mode() {
            return false;
        }
        (n <= DECODE_MAX_BATCH && parts.iter().all(|(m, _)| m.fmt != Fmt::Q8 || m.coarse || self.int8_q8))
            || (s.exact.get() && n <= VERIFY_MAX_BATCH && self.k.matmul_vfy.is_some() && parts.iter().all(|(m, _)| m.fmt != Fmt::Q8))
    }

    /// The dp4a matmuls on `n` tokens' activations already quantised into `s.xq` / `s.xd`. One token uses
    /// the 1-token kernels, batches the 8-token ones or, verifying, the tensor-core ones; all give a token
    /// bit-identical results.
    fn matmul_dp4a(&self, parts: &[(&GpuMatrix, usize)], out: u64, out_stride: usize, n: usize, accumulate: bool, s: &GpuSession) -> Result<(), String> {
        let k = &self.k;
        for &(m, off) in parts {
            if m.fmt == Fmt::Iq2 {
                k.launch(
                    k.matmul_iq2_dp4a,
                    (m.rows.div_ceil(8) as u32, n as u32),
                    256,
                    0,
                    &[Arg::P(m.q.ptr), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(out), Arg::I(m.rows as i32), Arg::I(m.cols as i32), Arg::I(accumulate as i32), Arg::I(out_stride as i32), Arg::I(off as i32), Arg::I(m.iq)],
                )?;
                continue;
            }
            if let Some(vfy) = k.matmul_vfy.filter(|_| !matches!(m.fmt, Fmt::Q8 | Fmt::Q3K) && n >= vfy_min() && (s.exact.get() || n > DECODE_MAX_BATCH)) {
                // 32 tokens per launch at most (the kernels' tiles and shared memory)
                let nb = m.cols / 32;
                for t0 in (0..n).step_by(32) {
                    let nt = (n - t0).min(32);
                    let func = vfy[(m.fmt == Fmt::Q6K) as usize][match nt.div_ceil(8) {
                        1 => 0,
                        2 => 1,
                        _ => 2,
                    }];
                    let (xq, xd, xs) = (s.xq.ptr + (t0 * m.cols) as u64, s.xd.ptr + (4 * t0 * nb) as u64, s.xsum.ptr + (4 * t0 * nb) as u64);
                    let out = out + (4 * t0 * out_stride) as u64;
                    k.launch(
                        func,
                        (m.rows.div_ceil(32) as u32, 1),
                        128,
                        0,
                        &[Arg::P(m.q.ptr), Arg::P(m.d_ptr()), Arg::P(xq), Arg::P(xd), Arg::P(xs), Arg::P(out), Arg::I(m.rows as i32), Arg::I(m.cols as i32), Arg::I(nt as i32), Arg::I(accumulate as i32), Arg::I(out_stride as i32), Arg::I(off as i32)],
                    )?;
                }
                continue;
            }
            let rpw = if m.fmt == Fmt::Q8 { 1 } else if n > 1 { batch_rpw() } else { kq_rpw() };
            let func = if m.fmt == Fmt::Q3K { k.matmul_q3k_dp4a[(n > 1) as usize] } else { k.matmul_dp4a[m.fmt as usize][(n > 1) as usize] };
            k.launch(
                func,
                (m.rows.div_ceil(8 * rpw) as u32, 1),
                256,
                0,
                &[Arg::P(m.q.ptr), Arg::P(m.d_ptr()), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.xsum.ptr), Arg::P(out), Arg::I(m.rows as i32), Arg::I(m.cols as i32), Arg::I(n as i32), Arg::I(accumulate as i32), Arg::I(out_stride as i32), Arg::I(off as i32)],
            )?;
        }
        Ok(())
    }

    /// RMSNorm of `x` into `s.xb`, then the matmul; for one token on the dp4a path the norm writes the
    /// int8 activations directly.
    fn norm_matmul(&self, norm: &DeviceBuf, parts: &[(&GpuMatrix, usize)], out: u64, out_stride: usize, n: usize, s: &GpuSession) -> Result<(), String> {
        let (k, c) = (&self.k, &self.cfg);
        if self.dp4a_path(parts, n, s) {
            k.launch(k.rmsnorm_q8, (n as u32, 1), 1024, 0, &[Arg::P(s.x.ptr), Arg::P(norm.ptr), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.xsum.ptr), Arg::I(c.dim as i32), Arg::F(c.norm_eps)])?;
            return self.matmul_dp4a(parts, out, out_stride, n, false, s);
        }
        k.launch(k.rmsnorm, (n as u32, 1), 256, 0, &[Arg::P(s.x.ptr), Arg::P(norm.ptr), Arg::P(s.xb.ptr), Arg::I(c.dim as i32), Arg::F(c.norm_eps)])?;
        self.matmul(parts, s.xb.ptr, out, out_stride, n, false, s)
    }

    fn matmul_one(&self, m: &GpuMatrix, x: u64, out: u64, n: usize, accumulate: bool, s: &GpuSession) -> Result<(), String> {
        self.matmul(&[(m, 0)], x, out, m.rows, n, accumulate, s)
    }

    /// The tiled verify-batch attention (scores and outputs kernels) for a launch of m tokens, and its tokens per
    /// block, when it applies: a verify batch, grouped-query heads of size 128 (at most 16 (token, head) pairs per
    /// block) and an int8 cache.
    fn vfy_attention(&self, s: &GpuSession, m: usize) -> Option<([Ptr; 2], usize)> {
        let c = &self.cfg;
        let funcs = self.k.attention_vfy?;
        let g = c.n_heads / c.n_kv_heads;
        if !s.exact.get() || m < 2 || !self.gqa_kernel() || !vfy_attention_fits(c) {
            return None;
        }
        let (i, nb) = match g {
            1 | 2 => (2, 8),
            3 | 4 => (1, 4),
            _ => (0, 2),
        };
        Some((funcs[i], nb))
    }

    /// The grouped-query attention kernel applies (head size a multiple of 32, at most 8 heads per kv head).
    fn gqa_kernel(&self) -> bool {
        let c = &self.cfg;
        c.head_dim % 32 == 0 && c.n_heads % c.n_kv_heads == 0 && c.n_heads / c.n_kv_heads <= ATT_MAX_G
    }

    /// Slices to aim for per token when generating: enough that a single token spreads over the GPU.
    /// Each token's slice size then follows from its own span alone (att_chunk_for in the kernels).
    fn att_want(&self) -> usize {
        ATT_TARGET_BLOCKS.div_ceil(if self.gqa_kernel() { self.cfg.n_kv_heads } else { self.cfg.n_heads })
    }

    /// Mirror of the kernels' att_chunk_for: short contexts in one block per head (no merge pass),
    /// longer ones split.
    fn att_chunk(span: usize, want: usize) -> usize {
        if span <= SPLIT_MIN_SPAN {
            return ATT_MAX_CHUNK;
        }
        span.div_ceil(want).clamp(32, att_chunk_cap())
    }

    /// Most slices any token of a ctx-position session needs: the fixed split grid of graph decoding.
    fn max_splits(&self, ctx: usize) -> usize {
        let want = self.att_want();
        (1..=ctx).map(|span| span.div_ceil(Self::att_chunk(span, want))).max().unwrap_or(1)
    }

    /// posp: device pointer to the position (CUDA graphs), else 0 and s.pos is used. ixed_split:
    /// one-token steps use the session's widest split grid, so a captured graph fits every position.
    /// kc, vc: the layer's cache; ks, vs: its int8 scales (unused with an f16 cache).
    fn attention(&self, s: &GpuSession, n: usize, (kc, vc, ks, vs): (u64, u64, u64, u64), posp: u64, fixed_split: bool) -> Result<(), String> {
        let c = &self.cfg;
        let k = &self.k;
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        let tc = k.attention_prefill_tc.and_then(|f| match c.head_dim {
            64 => Some(f[0]),
            128 => Some(f[1]),
            _ => None,
        });
        // a verify batch keeps every token on the split kernel below, as a one-token step
        let tiled = n >= TILED_MIN_BATCH && !s.exact.get();
        if let (true, Some(func)) = (tiled, tc) {
            return k.launch(
                func,
                (c.n_heads as u32, n.div_ceil(PREFILL_TC_QT) as u32),
                128,
                0,
                &[Arg::P(s.qkv.ptr), Arg::P(kc), Arg::P(vc), Arg::P(ks), Arg::P(vs), Arg::P(s.att.ptr), Arg::I(s.pos as i32), Arg::I(n as i32), Arg::I(c.n_heads as i32), Arg::I(c.n_kv_heads as i32), Arg::I(c.head_dim as i32), Arg::F(scale)],
            );
        }
        if let (true, Some(shared)) = (tiled, self.prefill_shared) {
            return k.launch(
                k.attention_prefill,
                (c.n_heads as u32, n.div_ceil(PREFILL_QT) as u32),
                128,
                shared as u32,
                &[Arg::P(s.qkv.ptr), Arg::P(kc), Arg::P(vc), Arg::P(ks), Arg::P(vs), Arg::P(s.att.ptr), Arg::I(s.pos as i32), Arg::I(n as i32), Arg::I(c.n_heads as i32), Arg::I(c.n_kv_heads as i32), Arg::I(c.head_dim as i32), Arg::F(scale)],
            );
        }
        // split-context path, a few tokens at a time so the partial-results buffer stays small
        let (q_dim, kv_dim) = (c.n_heads * c.head_dim, c.n_kv_heads * c.head_dim);
        let row = (q_dim + 2 * kv_dim) * 4;
        let mut b0 = 0;
        while b0 < n {
            let pos0 = s.pos + b0;
            let want = self.att_want();
            // SPLIT_MAX_BATCH tokens per launch, or the rest of a verify batch for the tiled kernel when the
            // partial results (sized for SPLIT_MAX_BATCH tokens' slices) hold them
            let mut m = (n - b0).min(SPLIT_MAX_BATCH);
            if s.exact.get() && self.k.attention_vfy.is_some() && vfy_attention_fits(c) && n - b0 > 1 {
                // the tiled kernels: as many tokens as their (token, head, slice) rows fit, in part and in scr
                let all = n - b0;
                let slices = (pos0 + 1..=pos0 + all).map(|span| span.div_ceil(Self::att_chunk(span, want))).max().unwrap_or(1);
                m = ((SPLIT_MAX_BATCH * s.cap.div_ceil(32)).min(AV_SCR_ROWS) / slices).clamp(1, all);
            }
            // the grid covers the token needing the most slices; the others' extra slices are empty
            let nsplit = if fixed_split {
                self.max_splits(s.cap)
            } else {
                (pos0 + 1..=pos0 + m).map(|span| span.div_ceil(Self::att_chunk(span, want))).max().unwrap_or(1)
            };
            let (qkv, out) = (s.qkv.at(b0 * row), s.att.at(b0 * q_dim * 4));
            if let Some(([scores, outputs], nb)) = self.vfy_attention(s, m) {
                // a verify batch: groups of nb tokens share every key and value row, each gets its one-token result
                let grid = (c.n_kv_heads as u32, nsplit as u32, m.div_ceil(nb) as u32);
                let (part, scr) = (s.att_part.ptr, s.att_scr.ptr);
                k.launch3(
                    scores,
                    grid,
                    256,
                    0,
                    &[Arg::P(qkv), Arg::P(kc), Arg::P(ks), Arg::P(vs), Arg::P(part), Arg::P(scr), Arg::I(pos0 as i32), Arg::I(c.n_heads as i32), Arg::I(c.n_kv_heads as i32), Arg::F(scale), Arg::I(nsplit as i32), Arg::I(want as i32), Arg::I(m as i32)],
                )?;
                k.launch3(
                    outputs,
                    grid,
                    128,
                    0,
                    &[Arg::P(vc), Arg::P(part), Arg::P(out), Arg::P(scr), Arg::I(pos0 as i32), Arg::I(c.n_heads as i32), Arg::I(c.n_kv_heads as i32), Arg::I(nsplit as i32), Arg::I(want as i32), Arg::I(m as i32)],
                )?;
            } else {
            let tc = k.attention_decode_tc256.filter(|_| c.head_dim == 256 && self.gqa_kernel());
            let (func, heads) = if let Some(f) = tc { (f, c.n_kv_heads) } else if self.gqa_kernel() { (k.attention_partial_gqa, c.n_kv_heads) } else { (k.attention_partial, c.n_heads) };
            k.launch3(
                func,
                (heads as u32, m as u32, nsplit as u32),
                128,
                0,
                &[Arg::P(qkv), Arg::P(kc), Arg::P(vc), Arg::P(ks), Arg::P(vs), Arg::P(s.att_part.ptr), Arg::P(out), Arg::I(pos0 as i32), Arg::P(posp), Arg::I(c.n_heads as i32), Arg::I(c.n_kv_heads as i32), Arg::I(c.head_dim as i32), Arg::F(scale), Arg::I(nsplit as i32), Arg::I(want as i32)],
            )?;
            }
            if nsplit > 1 {
                k.launch3(
                    k.attention_combine,
                    (c.n_heads as u32, m as u32, c.head_dim.div_ceil(32) as u32),
                    32,
                    0,
                    &[Arg::P(s.att_part.ptr), Arg::P(out), Arg::I(pos0 as i32), Arg::P(posp), Arg::I(c.n_heads as i32), Arg::I(c.head_dim as i32), Arg::I(nsplit as i32), Arg::I(want as i32)],
                )?;
            }
            b0 += m;
        }
        Ok(())
    }

    pub fn prefill(&self, tokens: &[u32], s: &mut GpuSession) -> Result<(), String> {
        // a mixture of experts takes short prompts a token at a time: one token runs the experts held in GPU
        // memory there, while a small batch would run every expert on the CPU, reading even the GPU-held
        // ones back from RAM (or, when RAM is short, from the disk)
        let per_token = self.cfg.is_moe() && tokens.len() > 1 && tokens.len() < MOE_GPU_MIN_BATCH;
        // room for the whole prompt at once (or less room, for a conversation much shorter than the last)
        self.fit_session(s, (s.pos + tokens.len()).min(s.ctx), true)?;
        let chunks: Vec<&[u32]> = if per_token { tokens.chunks(1).collect() } else { tokens.chunks(max_batch(&self.cfg)).collect() };
        PROGRESS.start(tokens.len());
        let mut result = Ok(());
        let mut done = 0;
        for (i, chunk) in chunks.iter().enumerate() {
            PROGRESS.base.store(done, Ordering::Relaxed);
            result = self.forward_batch(chunk, s, i + 1 == chunks.len());
            if result.is_err() {
                break;
            }
            done += chunk.len();
            PROGRESS.done.store(done, Ordering::Relaxed);
        }
        PROGRESS.total.store(0, Ordering::Relaxed);
        result
    }

    /// Runs `tokens` at positions `s.pos..`; copies next-token logits into `s.logits` if asked.
    pub fn forward_batch(&self, tokens: &[u32], s: &mut GpuSession, want_logits: bool) -> Result<(), String> {
        self.forward(Input::Tokens(tokens), s, if want_logits { Out::Last } else { Out::None }).map(|_| ())
    }

    /// Runs up to `verify_max()` `tokens` and returns the greedy next token after each of them (argmax on
    /// the GPU). Each position gets exactly the logits a one-token step would give it, so this is how a
    /// speculative draft is checked.
    pub fn forward_argmax(&self, tokens: &[u32], s: &mut GpuSession) -> Result<Vec<u32>, String> {
        if tokens.len() > self.verify_max() {
            return Err(format!("argmax batch of {} exceeds {}", tokens.len(), self.verify_max()));
        }
        self.forward(Input::Tokens(tokens), s, Out::Argmax)
    }

    /// Runs up to `verify_max()` `tokens` and copies the next-token logits after each of them into
    /// `s.rows` (row i at i * vocab): the same logits one-token steps would give, for checking a draft
    /// under sampling.
    pub fn forward_logits(&self, tokens: &[u32], s: &mut GpuSession) -> Result<(), String> {
        if tokens.len() > self.verify_max() {
            return Err(format!("logits batch of {} exceeds {}", tokens.len(), self.verify_max()));
        }
        self.forward(Input::Tokens(tokens), s, Out::All).map(|_| ())
    }

    /// Runs rows of `dim` floats (a model without an embedding table: CSM's transformers) at positions
    /// `s.pos..`. With `logits`, the last row's go to `s.logits`; either way its final-normed hidden state is
    /// kept for `hidden` and `head_logits`.
    pub fn forward_vectors(&self, xs: &[f32], s: &mut GpuSession, logits: bool) -> Result<(), String> {
        if !matches!(self.embd, Embd::Vectors) {
            return Err("this model reads tokens, not vectors".into());
        }
        if logits && self.output.is_none() {
            return Err("this model has no output layer (hidden states only)".into());
        }
        self.forward(Input::Vectors(xs), s, if logits { Out::Last } else { Out::Hidden }).map(|_| ())
    }

    /// The last pass's last row after the final norm (`dim` floats).
    pub fn hidden(&self, s: &GpuSession, out: &mut [f32]) -> Result<(), String> {
        let n = out.len().min(self.cfg.dim);
        self.k.d2h(&mut out[..n], s.xb.ptr)
    }

    /// `head` applied to the last pass's final-normed hidden state, into `s.logits[..head.rows()]`.
    pub fn head_logits(&self, head: &GpuHead, s: &mut GpuSession) -> Result<(), String> {
        let rows = head.0.rows;
        if rows > s.logits.len() || head.0.cols != self.cfg.dim {
            return Err(format!("a head of {rows} x {} does not fit this model", head.0.cols));
        }
        self.matmul_one(&head.0, s.xb.ptr, s.logits_dev.ptr, 1, false, s)?;
        self.k.d2h(&mut s.logits[..rows], s.logits_dev.ptr)
    }

    /// Runs this model on `other`'s kernels and stream (the same code: the compile options are global), so one
    /// CUDA graph can hold both models' passes (CsmFrames).
    pub fn share_kernels(&mut self, other: &GpuModel) -> Result<(), String> {
        // prompt attention's shared memory is a per-kernel limit: the larger of the two models' needs
        if let Some(bytes) = self.prefill_shared.filter(|&b| b > 48 * 1024 && other.prefill_shared.is_none_or(|o| b > o)) {
            let a = api()?;
            const MAX_DYNAMIC_SHARED: i32 = 8;
            if unsafe { (a.f.cuFuncSetAttribute)(other.k.attention_prefill, MAX_DYNAMIC_SHARED, bytes as i32) } != 0 {
                self.prefill_shared = None;
            }
        }
        self.k = other.k.clone();
        Ok(())
    }

    fn forward(&self, input: Input, s: &mut GpuSession, want: Out) -> Result<Vec<u32>, String> {
        let n = match input {
            Input::Tokens(t) => t.len(),
            Input::Vectors(xs) => xs.len() / self.cfg.dim,
        };
        if n == 0 {
            return Ok(Vec::new());
        }
        if n > max_batch(&self.cfg) {
            return Err(format!("GPU batch of {n} exceeds {}", max_batch(&self.cfg)));
        }
        if s.pos + n > s.ctx {
            return Err(format!("context full ({} tokens)", s.ctx));
        }
        if s.pos + n > s.cap {
            self.fit_session(s, s.pos + n, false)?;
        }
        let k = &self.k;
        let a = api()?;
        let tokens = match input {
            Input::Tokens(t) if matches!(self.embd, Embd::Vectors) => return Err(format!("this model reads vectors, not tokens ({} given)", t.len())),
            Input::Tokens(t) => t,
            Input::Vectors(xs) => {
                if xs.len() != n * self.cfg.dim {
                    return Err(format!("{} floats are not rows of {}", xs.len(), self.cfg.dim));
                }
                self.k.h2d(s.x.ptr, xs)?; // ordered before the pass's kernels
                &[]
            }
        };
        if !tokens.is_empty() {
            let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            self.k.h2d(s.tokens.ptr, &ids)?;
        }
        if let Embd::Host { map, offset, bytes, ty, rows, cols } = &self.embd {
            if let Some(&bad) = tokens.iter().find(|&&t| t as usize >= *rows) {
                return Err(format!("token id {bad} out of range"));
            }
            let m = crate::quant::Matrix { data: &map[*offset..*offset + *bytes], ty: *ty, rows: *rows, cols: *cols };
            let mut xs = vec![0f32; n * cols];
            for (&t, row) in tokens.iter().zip(xs.chunks_exact_mut(*cols)) {
                m.row_f32(t as usize, row);
            }
            self.k.h2d(s.x.ptr, &xs)?; // ordered before the pass's kernels
        }
        // checking a draft: every token as a one-token step would compute it
        s.exact.set(matches!(want, Out::Argmax | Out::All));
        // one-token steps replay a CUDA graph captured on the session's first one (GHOST_INFER_NO_GRAPH=1:
        // plain launches); the kernels then read the position from s.pos_dev
        // (not for a mixture of experts: its experts run on the CPU between the kernels)
        if n == 1 && want != Out::None && graphs_enabled() && !self.cfg.is_moe() {
            self.k.h2d(s.pos_dev.ptr, &[s.pos as i32])?;
            let slot = match want {
                Out::Argmax => 1,
                Out::Hidden => 2,
                _ => 0,
            };
            if s.graphs[slot].is_null() {
                const RELAXED: i32 = 2; // CU_STREAM_CAPTURE_MODE_RELAXED
                a.check(unsafe { (a.f.cuStreamBeginCapture_v2)(k.stream, RELAXED) }, "begin graph capture")?;
                let launched = self.launches(1, s, want, s.pos_dev.ptr, true);
                let mut graph: Ptr = std::ptr::null_mut();
                let ended = unsafe { (a.f.cuStreamEndCapture)(k.stream, &mut graph) };
                launched?;
                a.check(ended, "end graph capture")?;
                let mut exec: Ptr = std::ptr::null_mut();
                let made = unsafe { (a.f.cuGraphInstantiateWithFlags)(&mut exec, graph, 0) };
                unsafe { (a.f.cuGraphDestroy)(graph) };
                a.check(made, "instantiate graph")?;
                s.graphs[slot] = exec;
            }
            a.check(unsafe { (a.f.cuGraphLaunch)(s.graphs[slot], k.stream) }, "graph launch")?;
        } else {
            self.launches(n, s, want, 0, false)?;
        }
        let mut picks = Vec::new();
        if want == Out::Last {
            self.k.d2h(&mut s.logits, s.logits_dev.ptr)?;
        } else if want == Out::All {
            s.rows.resize(n * s.logits.len(), 0.0);
            self.k.d2h(&mut s.rows, s.logits_dev.ptr)?;
        } else if want == Out::Argmax {
            let mut ids = vec![0i32; n];
            self.k.d2h(&mut ids, s.picks.ptr)?;
            picks = ids.into_iter().map(|i| i as u32).collect();
        }
        s.pos += n;
        if let Some(cache) = self.cache.borrow_mut().as_mut() {
            cache.tick(n);
        }
        Ok(picks)
    }

    /// Mixture-of-experts feed-forward of layer `l`. The normalised rows come down to the CPU, which
    /// routes them. For one token, the experts held in GPU memory run there (queued first) while the CPU
    /// computes the others; then both are added to s.x. Prompt batches run every expert on the CPU.
    fn moe_ffn(&self, l: usize, layer: &GpuLayer, n: usize, s: &GpuSession) -> Result<(), String> {
        let (c, k, a) = (&self.cfg, &self.k, api()?);
        let moe = self.moe.as_ref().ok_or("mixture of experts without its experts")?;
        let dim = c.dim;
        let mut cache_ref = self.cache.borrow_mut();
        if cache_ref.is_none() {
            *cache_ref = Some(ExpertCache::new(moe, c, expert_cache_budget()?)?);
        }
        let cache = cache_ref.as_mut().unwrap();
        if n >= MOE_GPU_MIN_BATCH && cache.batch_ok {
            return self.moe_ffn_batch(l, layer, n, s, cache, moe);
        }
        let on_gpu = n == 1 && cache.cacheable[l] && cache.slots() > 0;
        k.launch(k.rmsnorm, (n as u32, 1), 256, 0, &[Arg::P(s.x.ptr), Arg::P(layer.ffn_norm.ptr), Arg::P(s.xb.ptr), Arg::I(dim as i32), Arg::F(c.norm_eps)])?;
        if on_gpu {
            k.launch(k.rmsnorm_q8, (1, 1), 1024, 0, &[Arg::P(s.x.ptr), Arg::P(layer.ffn_norm.ptr), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.xsum.ptr), Arg::I(dim as i32), Arg::F(c.norm_eps)])?;
        }
        // one token's router logits come from the GPU, right after the normalised row (s.xb holds a batch)
        let ne = c.n_experts;
        let gpu_route = n == 1 && dim % 4 == 0;
        if gpu_route {
            k.launch(k.router_logits1, (ne.div_ceil(2) as u32, 1), 64, 0, &[Arg::P(s.xb.ptr), Arg::P(cache.routers[l].ptr), Arg::P(s.xb.ptr + (dim * 4) as u64), Arg::I(dim as i32), Arg::I(ne as i32)])?;
        }
        let mut guard = s.moe.borrow_mut();
        let h = &mut *guard;
        let len = n * dim;
        let got = if gpu_route { len + ne } else { len };
        h.xb.resize(got, 0.0);
        let stats = moe_stats_on() && on_gpu;
        let t0 = stats.then(Instant::now);
        // synchronous copy: waits for the norm (the launch stream is a blocking one)
        a.check(unsafe { (a.f.cuMemcpyDtoH_v2)(h.xb.as_mut_ptr() as *mut c_void, s.xb.ptr, got * 4) }, "copy to host")?;
        let t1 = stats.then(Instant::now);
        if gpu_route {
            let logits = &h.xb[len..];
            if c.mla.is_some() {
                crate::model::route_deepseek(logits, moe.biases[l].as_deref(), c.n_experts_used, c.gating_sigmoid, c.expert_weights_norm, c.expert_weights_scale, &mut h.picks);
            } else {
                crate::model::route(logits, c.n_experts_used, &mut h.picks);
            }
            h.xb.truncate(len);
        } else if c.mla.is_some() {
            crate::model::moe_route_deepseek(c, &moe.routers[l], moe.biases[l].as_deref(), &h.xb, &mut h.picks, &mut h.scratch);
        } else {
            crate::model::moe_route(&moe.routers[l], c.n_experts_used, &h.xb, dim, &mut h.picks, &mut h.scratch);
        }
        let t2 = stats.then(Instant::now);

        // the token's GPU-resident experts: table of matrix addresses, then their weights
        let mut ptrs: Vec<u64> = Vec::new();
        let mut downs: Vec<u64> = Vec::new();
        let mut wts: Vec<f32> = Vec::new();
        h.cpu_picks.clear();
        for &(e, w) in &h.picks {
            match cache.slot_of[l][e] {
                slot if on_gpu && slot >= 0 => {
                    let [g, u, d] = cache.ptrs(slot as usize, l);
                    ptrs.extend([g, u]);
                    downs.push(d);
                    wts.push(w);
                }
                _ => h.cpu_picks.push((e, w)),
            }
        }
        let nk = wts.len();
        if nk > 0 {
            ptrs.extend(&downs);
            let mut tab: Vec<u8> = ptrs.iter().flat_map(|p| p.to_le_bytes()).collect();
            tab.extend(wts.iter().flat_map(|w| w.to_le_bytes()));
            s.moe_tab.write(&tab)?;
            let (ex, ffn) = (moe.tensors[l][0], c.expert_ffn_dim);
            let silu = [Arg::P(s.gu.ptr), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.xsum.ptr), Arg::I(ffn as i32)];
            match iq2_fmt(&moe.tensors[l]) {
                Some(fmt) => {
                    k.launch(k.moe_gate_up_iq2, (ex.rows.div_ceil(8) as u32, (2 * nk) as u32), 256, 0, &[Arg::P(s.moe_tab.ptr), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.gu.ptr), Arg::I(ffn as i32), Arg::I(dim as i32), Arg::I(fmt)])?;
                    k.launch(k.silu_q8, ((ffn / 4).div_ceil(256) as u32, nk as u32), 256, 0, &silu)?;
                    k.launch(k.moe_down_iq2, (dim.div_ceil(8) as u32, nk as u32), 256, 0, &[Arg::P(s.moe_tab.ptr), Arg::I(nk as i32), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.h.ptr), Arg::I(dim as i32), Arg::I(ffn as i32), Arg::I(fmt)])?;
                }
                None => {
                    k.launch(k.moe_gate_up_q3k, (ex.rows.div_ceil(8) as u32, (2 * nk) as u32), 256, 0, &[Arg::P(s.moe_tab.ptr), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.xsum.ptr), Arg::P(s.gu.ptr), Arg::I(ffn as i32), Arg::I(dim as i32)])?;
                    k.launch(k.silu_q8, ((ffn / 4).div_ceil(256) as u32, nk as u32), 256, 0, &silu)?;
                    k.launch(k.moe_down_q4k, (dim.div_ceil(8) as u32, nk as u32), 256, 0, &[Arg::P(s.moe_tab.ptr), Arg::I(nk as i32), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.xsum.ptr), Arg::P(s.h.ptr), Arg::I(dim as i32), Arg::I(ffn as i32)])?;
                }
            }
        }
        // the CPU's share, computed while the GPU works on its own
        let cpu_k = if on_gpu { h.cpu_picks.len() } else { c.n_experts_used };
        let cpu_picks = if on_gpu { &h.cpu_picks } else { &h.picks };
        let mut t3 = None;
        let cpu = if cpu_k > 0 {
            h.acc.clear();
            h.acc.resize(len, 0.0);
            crate::model::moe_experts(c.n_experts, |e| moe.expert(l, e), cpu_picks, cpu_k, &h.xb, dim, &mut h.acc, &mut h.scratch);
            t3 = stats.then(Instant::now);
            s.att.write(&h.acc)?; // free here: the attention output was consumed by the output projection
            s.att.ptr
        } else {
            0
        };
        let t4 = stats.then(Instant::now);
        // GHOST_INFER_MOE_CHECK=1: the GPU-resident experts again on the CPU, largest difference printed
        if nk > 0 && std::env::var_os("GHOST_INFER_MOE_CHECK").is_some() {
            let mut gpu = vec![0f32; nk * dim];
            a.check(unsafe { (a.f.cuMemcpyDtoH_v2)(gpu.as_mut_ptr() as *mut c_void, s.h.ptr, nk * dim * 4) }, "copy to host")?;
            let mut i = 0;
            for &(e, _) in &h.picks {
                if cache.slot_of[l][e] < 0 {
                    continue;
                }
                let mut want = vec![0f32; dim];
                crate::model::moe_experts(c.n_experts, |e| moe.expert(l, e), &[(e, 1.0)], 1, &h.xb, dim, &mut want, &mut h.scratch);
                let got = &gpu[i * dim..(i + 1) * dim];
                let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-20);
                let err = got.iter().zip(&want).fold(0f32, |m, (g, w)| m.max((g - w).abs())) / scale;
                eprintln!("[moe check layer {l} expert {e}: max |gpu - cpu| / max |cpu| = {err:.2e}]");
                i += 1;
            }
        }
        if nk > 0 {
            let wts_ptr = s.moe_tab.ptr + (8 * 3 * nk) as u64;
            k.launch(k.moe_combine, (dim.div_ceil(256) as u32, 1), 256, 0, &[Arg::P(s.x.ptr), Arg::P(s.h.ptr), Arg::P(wts_ptr), Arg::I(nk as i32), Arg::P(cpu), Arg::I(dim as i32)])?;
        } else {
            k.launch(k.add_rows, (len.div_ceil(256) as u32, 1), 256, 0, &[Arg::P(s.x.ptr), Arg::P(cpu), Arg::I(len as i32)])?;
        }
        cache.note(moe, l, &h.picks, false);
        if let (Some(t0), Some(t1), Some(t2), Some(t4)) = (t0, t1, t2, t4) {
            let ns = |a: Instant, b: Instant| (b - a).as_nanos() as u64;
            let t3 = t3.unwrap_or(t4);
            moe_stat(MS_HITS, nk as u64);
            moe_stat(MS_MISSES, h.cpu_picks.len() as u64);
            moe_stat(MS_STEPS, 1);
            moe_stat(MS_WAIT, ns(t0, t1));
            moe_stat(MS_ROUTE, ns(t1, t2));
            moe_stat(MS_CPU, ns(t2, t3));
            moe_stat(MS_BACK, ns(t3, t4));
            moe_stat(MS_NOTE, ns(t4, Instant::now()));
        }
        Ok(())
    }

    /// Mixture of experts for a prompt batch, all on the GPU: routed there (f32), grouped by expert on the
    /// host, then per expert: its tokens gathered, [gate | up] (converted to Q8) and down applied in
    /// batched matmuls, and the weighted results added back to their tokens. Experts not in the cache
    /// are staged through pinned memory, overlapping the previous expert's maths.
    fn moe_ffn_batch(&self, l: usize, layer: &GpuLayer, n: usize, s: &GpuSession, cache: &mut ExpertCache, moe: &CpuMoe) -> Result<(), String> {
        let (c, k, a) = (&self.cfg, &self.k, api()?);
        let (dim, ne, top, ffn) = (c.dim, c.n_experts, c.n_experts_used, c.expert_ffn_dim);
        // a batch uses nearly every expert of every layer: the prefetch thread reads the next layer's uncached
        // ones in (and this one's, at the first layer) while this layer computes. This layer waits for its
        // own read to finish first: faulting the same weights in at the same time would make a hard disk
        // seek between the two and read both slowly.
        if l == 0 {
            moe.prefetch_layer(0, &cache.slot_of[0]);
        }
        if l + 1 < c.n_layers {
            moe.prefetch_layer(l + 1, &cache.slot_of[l + 1]);
        }
        let waited = moe.prefetch.wait(l);
        if waited.as_millis() >= 20 {
            PROGRESS.waited_on_disk(waited.as_millis() as u64);
        }
        k.launch(k.rmsnorm, (n as u32, 1), 256, 0, &[Arg::P(s.x.ptr), Arg::P(layer.ffn_norm.ptr), Arg::P(s.xb.ptr), Arg::I(dim as i32), Arg::F(c.norm_eps)])?;
        k.launch(k.router_logits, (n as u32, 1), 256, (dim * 4) as u32, &[Arg::P(s.xb.ptr), Arg::P(cache.routers[l].ptr), Arg::P(s.gu.ptr), Arg::I(dim as i32), Arg::I(ne as i32)])?;
        let mut guard = s.moe.borrow_mut();
        let h = &mut *guard;
        h.xb.resize(n * ne, 0.0); // the logits
        a.check(unsafe { (a.f.cuMemcpyDtoH_v2)(h.xb.as_mut_ptr() as *mut c_void, s.gu.ptr, n * ne * 4) }, "copy router logits")?;
        h.picks.clear();
        for row in h.xb.chunks_exact(ne) {
            if c.mla.is_some() {
                crate::model::route_deepseek(row, moe.biases[l].as_deref(), top, c.gating_sigmoid, c.expert_weights_norm, c.expert_weights_scale, &mut h.cpu_picks);
            } else {
                crate::model::route(row, top, &mut h.cpu_picks);
            }
            h.picks.extend_from_slice(&h.cpu_picks);
        }
        cache.note(moe, l, &h.picks, true);

        // each expert's tokens, then all their weights, in one table
        let mut lists: Vec<Vec<(i32, f32)>> = vec![Vec::new(); ne];
        for (i, &(e, w)) in h.picks.iter().enumerate() {
            lists[e].push(((i / top) as i32, w));
        }
        let mut tab: Vec<u8> = Vec::with_capacity(n * top * 8);
        let mut offs = Vec::with_capacity(ne);
        for list in &lists {
            offs.push(tab.len() / 4);
            tab.extend(list.iter().flat_map(|(t, _)| t.to_le_bytes()));
        }
        let w_base = tab.len() / 4;
        for list in &lists {
            tab.extend(list.iter().flat_map(|(_, w)| w.to_le_bytes()));
        }
        cache.idx.write(&tab)?;

        let [g, _, d] = moe.tensors[l];
        let fmt = iq2_fmt(&moe.tensors[l]);
        let gate_blocks = g.rows * g.cols / 32;
        for (e, list) in lists.iter().enumerate().filter(|(_, list)| !list.is_empty()) {
            let m = list.len();
            let (ptrs, staged) = match cache.slot_of[l][e] {
                slot if slot >= 0 => (cache.ptrs(slot as usize, l), None),
                _ => {
                    let (p, i) = cache.stage(moe, l, e, k.stream)?;
                    (p, Some(i))
                }
            };
            // gate and up to Q8, one after the other: a single [gate | up] matrix of 2 * ffn rows
            for (j, &p) in ptrs[..2].iter().enumerate() {
                let (q, dq) = (cache.conv_q.ptr + (j * gate_blocks * 32) as u64, cache.conv_d.ptr + (j * gate_blocks * 4) as u64);
                match fmt {
                    Some(f) => k.launch(k.iq2_to_q8, (gate_blocks.div_ceil(8) as u32, 1), 256, 0, &[Arg::P(p), Arg::P(q), Arg::P(dq), Arg::I(gate_blocks as i32), Arg::I(f)])?,
                    None => k.launch(k.q3k_to_q8, (gate_blocks.div_ceil(8) as u32, 1), 256, 0, &[Arg::P(p), Arg::P(q), Arg::P(dq), Arg::I(gate_blocks as i32)])?,
                }
            }
            let idx = cache.idx.ptr + (offs[e] * 4) as u64;
            let wts = cache.idx.ptr + ((w_base + offs[e]) * 4) as u64;
            k.launch(k.moe_gather, (m as u32, 1), 256, 0, &[Arg::P(s.xb.ptr), Arg::P(idx), Arg::P(s.att.ptr), Arg::I(dim as i32)])?;
            let gate_up = GpuMatrix::view(Fmt::Q8, cache.conv_q.ptr, Some(cache.conv_d.ptr), 2 * ffn, dim);
            self.matmul(&[(&gate_up, 0)], s.att.ptr, s.gu.ptr, 2 * ffn, m, false, s)?;
            k.launch(k.silu_mul, ((m * ffn).div_ceil(256) as u32, 1), 256, 0, &[Arg::P(s.gu.ptr), Arg::P(s.h.ptr), Arg::I(ffn as i32), Arg::I(m as i32)])?;
            let down = match d.ty {
                _ if fmt.is_some() => {
                    let blocks = dim * ffn / 32;
                    k.launch(k.iq2_to_q8, (blocks.div_ceil(8) as u32, 1), 256, 0, &[Arg::P(ptrs[2]), Arg::P(cache.conv_q.ptr), Arg::P(cache.conv_d.ptr), Arg::I(blocks as i32), Arg::I(fmt.unwrap())])?;
                    GpuMatrix::view(Fmt::Q8, cache.conv_q.ptr, Some(cache.conv_d.ptr), dim, ffn)
                }
                GgmlType::Q4K => GpuMatrix::view(Fmt::Q4K, ptrs[2], None, dim, ffn),
                GgmlType::Q5K => {
                    let blocks = dim * ffn / 32;
                    k.launch(k.q5k_to_q8, (blocks.div_ceil(8) as u32, 1), 256, 0, &[Arg::P(ptrs[2]), Arg::P(cache.conv_q.ptr), Arg::P(cache.conv_d.ptr), Arg::I(blocks as i32)])?;
                    GpuMatrix::view(Fmt::Q8, cache.conv_q.ptr, Some(cache.conv_d.ptr), dim, ffn)
                }
                other => return Err(format!("GPU experts: {} down projections are not supported yet", other.name())),
            };
            self.matmul(&[(&down, 0)], s.h.ptr, s.qkv.ptr, dim, m, false, s)?;
            k.launch(k.moe_scatter, (m as u32, 1), 256, 0, &[Arg::P(s.x.ptr), Arg::P(s.qkv.ptr), Arg::P(idx), Arg::P(wts), Arg::I(dim as i32)])?;
            if let Some(i) = staged {
                cache.release(i, k.stream)?;
            }
        }
        PROGRESS.layer(l, c.n_layers, n);
        Ok(())
    }

    /// A Gated DeltaNet layer (see model::Delta): the [q k v | z | ba] projection of the normed rows into s.gu,
    /// the convolution, unit q and k with the gates (beta, decay) in s.h, the recurrence through the layer's
    /// state into s.att, and the gated norm there, ready for ssm_out.
    fn delta_net(&self, l: usize, layer: &GpuLayer, d: &GpuDelta, n: usize, s: &GpuSession) -> Result<(), String> {
        let (c, k) = (&self.cfg, &self.k);
        let dc = c.delta.unwrap();
        let (cd, vd) = (dc.conv_dim(), dc.value_dim());
        let stride = cd + vd + 2 * dc.v_heads;
        let proj: Vec<(&GpuMatrix, usize)> = d.proj.iter().map(|(m, off)| (m, *off)).collect();
        self.norm_matmul(&layer.attn_norm, &proj, s.gu.ptr, stride, n, s)?;
        let (conv_state, state) = s.delta_state(c, c.slot(l));
        k.launch(k.delta_conv, (cd.div_ceil(256) as u32, 1), 256, 0, &[Arg::P(s.gu.ptr), Arg::I(stride as i32), Arg::P(d.conv.ptr), Arg::P(conv_state), Arg::I(n as i32), Arg::I(cd as i32), Arg::I(dc.conv as i32 - 1)])?;
        let qscale = 1.0 / (dc.head as f32).sqrt();
        k.launch(
            k.delta_prep,
            (n as u32, (2 * dc.k_heads + 1) as u32),
            dc.head as u32,
            0,
            &[Arg::P(s.gu.ptr), Arg::I(stride as i32), Arg::I((cd + vd) as i32), Arg::P(d.dt_bias.ptr), Arg::P(d.a.ptr), Arg::P(s.h.ptr), Arg::I(dc.k_heads as i32), Arg::I(dc.v_heads as i32), Arg::I(dc.head as i32), Arg::F(qscale), Arg::F(c.norm_eps), Arg::I(dc.tiled as i32)],
        )?;
        k.launch(k.delta_scan, (dc.v_heads as u32, (dc.head / 32) as u32), 32, 0, &[Arg::P(s.gu.ptr), Arg::I(stride as i32), Arg::P(s.h.ptr), Arg::P(state), Arg::P(s.att.ptr), Arg::I(n as i32), Arg::I(dc.k_heads as i32), Arg::I(dc.v_heads as i32), Arg::I(dc.tiled as i32)])?;
        k.launch(k.delta_norm, (n as u32, dc.v_heads as u32), dc.head as u32, 0, &[Arg::P(s.att.ptr), Arg::P(s.gu.ptr), Arg::I(stride as i32), Arg::I(cd as i32), Arg::P(d.norm.ptr), Arg::I(dc.v_heads as i32), Arg::I(dc.head as i32), Arg::F(c.norm_eps)])
    }

    /// Layer l's feed-forward, added to s.x.
    fn ffn(&self, l: usize, layer: &GpuLayer, n: usize, s: &GpuSession) -> Result<(), String> {
        match &layer.ffn {
            GpuFfn::Dense { gate_up, down } => self.swiglu(gate_up, down, &layer.ffn_norm, None, n, s),
            GpuFfn::Moe { shared, gate } => {
                self.moe_ffn(l, layer, n, s)?;
                match (shared, gate) {
                    // the shared expert reads the same normed input, still in s.xb; a gated one goes to s.att
                    // first (free again: the experts' sums have been added), then in through its gate
                    (Some((gate_up, down)), Some(g)) => {
                        self.swiglu_to(gate_up, down, &layer.ffn_norm, Some(s.xb.ptr), n, s, s.att.ptr, false)?;
                        let k = &self.k;
                        k.launch(k.gated_add, (n as u32, 1), 256, 0, &[Arg::P(s.x.ptr), Arg::P(s.att.ptr), Arg::P(g.ptr), Arg::P(s.xb.ptr), Arg::I(self.cfg.dim as i32)])
                    }
                    (Some((gate_up, down)), None) => self.swiglu(gate_up, down, &layer.ffn_norm, Some(s.xb.ptr), n, s),
                    (None, _) => Ok(()),
                }
            }
        }
    }

    /// A SwiGLU block added to s.x: [gate | up] from RMSNorm(s.x) with `norm` (or from `normed`, already
    /// normed input), then down.
    fn swiglu(&self, gate_up: &[(GpuMatrix, usize)], down: &GpuMatrix, norm: &DeviceBuf, normed: Option<u64>, n: usize, s: &GpuSession) -> Result<(), String> {
        self.swiglu_to(gate_up, down, norm, normed, n, s, s.x.ptr, true)
    }

    /// `swiglu` into `out` (n rows of dim), added to it if `accumulate`.
    #[allow(clippy::too_many_arguments)]
    fn swiglu_to(&self, gate_up: &[(GpuMatrix, usize)], down: &GpuMatrix, norm: &DeviceBuf, normed: Option<u64>, n: usize, s: &GpuSession, out: u64, accumulate: bool) -> Result<(), String> {
        let k = &self.k;
        let hidden = down.cols;
        let gate_up: Vec<(&GpuMatrix, usize)> = gate_up.iter().map(|(m, off)| (m, *off)).collect();
        match normed {
            Some(x) => self.matmul(&gate_up, x, s.gu.ptr, 2 * hidden, n, false, s)?,
            None => self.norm_matmul(norm, &gate_up, s.gu.ptr, 2 * hidden, n, s)?,
        }
        let down = [(down, 0)];
        if self.dp4a_path(&down, n, s) {
            // decoding: silu(gate) * up straight to int8 for the down projection
            let nv = (hidden / 4) as u32;
            k.launch(k.silu_q8, (nv.div_ceil(256), n as u32), 256, 0, &[Arg::P(s.gu.ptr), Arg::P(s.xq.ptr), Arg::P(s.xd.ptr), Arg::P(s.xsum.ptr), Arg::I(hidden as i32)])?;
            return self.matmul_dp4a(&down, out, self.cfg.dim, n, accumulate, s);
        }
        let total = (n * hidden) as u32;
        k.launch(k.silu_mul, (total.div_ceil(256), 1), 256, 0, &[Arg::P(s.gu.ptr), Arg::P(s.h.ptr), Arg::I(hidden as i32), Arg::I(n as i32)])?;
        self.matmul_one(down[0].0, s.h.ptr, out, n, accumulate, s)
    }

    /// Latent attention's inputs: the queries (low-rank step if any) to s.qkv, the compressed key-value
    /// rows normed and rotated into the cache, and each head's query moved into the latent space, its RoPE
    /// part rotated after it, in s.gu (heads x (kl + rd) per token).
    fn mla_in(&self, layer: &GpuLayer, mla: &GpuMla, cache: u64, n: usize, s: &GpuSession, posp: u64) -> Result<(), String> {
        let (c, k) = (&self.cfg, &self.k);
        let m = c.mla.unwrap();
        let (h, hd, rd, kd) = (c.n_heads, c.head_dim, c.rope_dim, c.k_row());
        let q_dim = h * hd;
        // s.qkv: the queries (max_batch rows of q_dim), then rows of [low-rank query | compressed kv]
        let (q, lat) = (s.qkv.ptr, s.qkv.at(max_batch(c) * q_dim * 4));
        let lat_w = m.q_lora + kd;
        match &mla.q_a {
            Some((wa, wa_norm)) => {
                self.norm_matmul(&layer.attn_norm, &[(wa, 0), (&mla.kv_a, m.q_lora)], lat, lat_w, n, s)?;
                k.launch(k.rmsnorm_rows, (n as u32, 1), 256, 0, &[Arg::P(lat), Arg::I(lat_w as i32), Arg::P(wa_norm.ptr), Arg::P(s.xb.ptr), Arg::I(m.q_lora as i32), Arg::I(m.q_lora as i32), Arg::F(c.norm_eps)])?;
                self.matmul_one(&mla.q, s.xb.ptr, q, n, false, s)?;
            }
            None => {
                self.norm_matmul(&layer.attn_norm, &[(&mla.kv_a, 0)], lat, lat_w, n, s)?;
                self.norm_matmul(&layer.attn_norm, &[(&mla.q, 0)], q, q_dim, n, s)?;
            }
        }
        k.launch(
            k.mla_kv,
            (n as u32, 1),
            256,
            0,
            &[Arg::P(lat), Arg::I(lat_w as i32), Arg::I(m.q_lora as i32), Arg::P(mla.kv_norm.ptr), Arg::P(cache), Arg::I(s.pos as i32), Arg::P(posp), Arg::I(m.kv_lora as i32), Arg::I(rd as i32), Arg::F(c.rope_base), Arg::F(c.norm_eps)],
        )?;
        let qa = s.gu.ptr;
        k.launch3(
            k.head_matvec,
            (n.div_ceil(HMV_T) as u32, h as u32, hmv_slices(n, m.kv_lora) as u32),
            256,
            (HMV_T * m.nope * 4) as u32,
            &[Arg::P(mla.k_b.ptr), Arg::P(q), Arg::P(qa), Arg::I(m.kv_lora as i32), Arg::I(m.nope as i32), Arg::I(n as i32), Arg::I(q_dim as i32), Arg::I(hd as i32), Arg::I((h * kd) as i32), Arg::I(kd as i32)],
        )?;
        k.launch(k.mla_q_rope, (n as u32, h as u32), 32, 0, &[Arg::P(q), Arg::P(qa), Arg::I(s.pos as i32), Arg::P(posp), Arg::I(hd as i32), Arg::I(m.kv_lora as i32), Arg::I(rd as i32), Arg::F(c.rope_base)])
    }

    /// Latent attention over the layer's cache (queries in s.gu) into s.att (heads x kl per token).
    /// fixed_split: one-token steps use the session's widest split grid, so a captured graph fits.
    fn mla_attention(&self, cache: u64, n: usize, s: &GpuSession, posp: u64, fixed_split: bool) -> Result<(), String> {
        let (c, k) = (&self.cfg, &self.k);
        let m = c.mla.unwrap();
        let (h, rd) = (c.n_heads, c.rope_dim);
        let span = if fixed_split { s.cap } else { s.pos + n };
        let nsplit = if n > MLA_SPLIT_MAX_BATCH { 1 } else { span.div_ceil(MLA_SPLIT_SPAN).clamp(1, MLA_MAX_SPLIT) };
        let shared = (h * (m.kv_lora + rd) + h * MLA_T + 3 * h) * 4;
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        k.launch(
            k.mla_attention,
            (n as u32, nsplit as u32),
            256,
            shared as u32,
            &[Arg::P(s.gu.ptr), Arg::P(cache), Arg::P(s.att_part.ptr), Arg::P(s.att.ptr), Arg::I(s.pos as i32), Arg::P(posp), Arg::I(h as i32), Arg::I(m.kv_lora as i32), Arg::I(rd as i32), Arg::F(scale), Arg::I(nsplit as i32)],
        )?;
        if nsplit > 1 {
            k.launch(k.mla_combine, (n as u32, h as u32), 256, 0, &[Arg::P(s.att_part.ptr), Arg::P(s.att.ptr), Arg::I(h as i32), Arg::I(m.kv_lora as i32), Arg::I(nsplit as i32)])?;
        }
        Ok(())
    }

    /// Each head's latent result out to its values (v_b, into s.qkv: the queries are used up), then the
    /// output projection added to s.x.
    fn mla_out(&self, layer: &GpuLayer, mla: &GpuMla, n: usize, s: &GpuSession) -> Result<(), String> {
        let (c, k) = (&self.cfg, &self.k);
        let m = c.mla.unwrap();
        let h = c.n_heads;
        k.launch3(
            k.head_matvec,
            (n.div_ceil(HMV_T) as u32, h as u32, hmv_slices(n, m.v_head) as u32),
            256,
            (HMV_T * m.kv_lora * 4) as u32,
            &[Arg::P(mla.v_b.ptr), Arg::P(s.att.ptr), Arg::P(s.qkv.ptr), Arg::I(m.v_head as i32), Arg::I(m.kv_lora as i32), Arg::I(n as i32), Arg::I((h * m.kv_lora) as i32), Arg::I(m.kv_lora as i32), Arg::I((h * m.v_head) as i32), Arg::I(m.v_head as i32)],
        )?;
        self.matmul_one(&layer.wo, s.qkv.ptr, s.x.ptr, n, true, s)
    }

    /// Every kernel of a forward pass over `n` tokens at `s.pos..` (`posp`: the position from device memory
    /// instead, for graphs), ending with the last token's logits (Last) or every token's argmax (Argmax) on
    /// the device. Launches only, so it can be captured.
    fn launches(&self, n: usize, s: &GpuSession, want: Out, posp: u64, fixed_split: bool) -> Result<(), String> {
        let c = &self.cfg;
        let (dim, hd) = (c.dim, c.head_dim);
        let k = &self.k;

        // profile buckets match profile::OPS; with profiling off `timed` is a plain call
        // (an embedding left in the file: forward has copied the rows into s.x already)
        if let Embd::Gpu(e) = &self.embd {
            self.timed(0, || k.launch(k.embed, (n as u32, 1), 256, 0, &[Arg::I(e.fmt as i32), Arg::P(e.q.ptr), Arg::P(e.d_ptr()), Arg::P(s.tokens.ptr), Arg::P(s.x.ptr), Arg::I(dim as i32)]))?;
        }
        let qkv_rows = (c.n_heads + 2 * c.n_kv_heads) * hd;
        for (l, layer) in self.layers.iter().enumerate() {
            if let Some(mla) = &layer.mla {
                let cache = s.k_cache.at(l);
                self.timed(2, || self.mla_in(layer, mla, cache, n, s, posp))?;
                self.timed(4, || self.mla_attention(cache, n, s, posp, fixed_split))?;
                self.timed(5, || self.mla_out(layer, mla, n, s))?;
                self.timed(6, || self.ffn(l, layer, n, s))?;
                continue;
            }
            if let Some(d) = &layer.delta {
                self.timed(2, || self.delta_net(l, layer, d, n, s))?;
                self.timed(5, || self.matmul_one(&layer.wo, s.att.ptr, s.x.ptr, n, true, s))?;
                self.timed(6, || self.ffn(l, layer, n, s))?;
                continue;
            }
            // the caches hold attention layers only (a DeltaNet layer keeps a state instead)
            let slot = c.slot(l);
            let (kc, vc) = (s.k_cache.at(slot), s.v_cache.at(slot));
            let (ks, vs) = (s.k_scale.at(slot), s.v_scale.at(slot));
            let qkv: Vec<(&GpuMatrix, usize)> = layer.qkv.iter().map(|(m, off)| (m, *off)).collect();
            if c.attn_gated {
                // [q | gate] per head, then k and v, into s.gu; split into the packed rows and the gates (s.h)
                let rows = 2 * c.n_heads * hd + 2 * c.n_kv_heads * hd;
                self.timed(2, || {
                    self.norm_matmul(&layer.attn_norm, &qkv, s.gu.ptr, rows, n, s)?;
                    k.launch(k.next_split_qgate, (n as u32, (c.n_heads + 2 * c.n_kv_heads) as u32), 128, 0, &[Arg::P(s.gu.ptr), Arg::I(rows as i32), Arg::P(s.qkv.ptr), Arg::P(s.h.ptr), Arg::I(c.n_heads as i32), Arg::I(c.n_kv_heads as i32), Arg::I(hd as i32)])
                })?;
            } else {
                self.timed(2, || self.norm_matmul(&layer.attn_norm, &qkv, s.qkv.ptr, qkv_rows, n, s))?;
            }
            self.timed(3, || {
                let (qn, kn) = match (&layer.q_norm, &layer.k_norm) {
                    (Some(qn), Some(kn)) => (qn.ptr, kn.ptr),
                    _ => (0, 0),
                };
                k.launch(
                    k.qk_norm_rope,
                    (n as u32, (c.n_heads + 2 * c.n_kv_heads) as u32),
                    64,
                    0,
                    &[Arg::P(s.qkv.ptr), Arg::P(qn), Arg::P(kn), Arg::P(kc), Arg::P(vc), Arg::P(ks), Arg::P(vs), Arg::I(s.pos as i32), Arg::P(posp), Arg::I(c.n_heads as i32), Arg::I(c.n_kv_heads as i32), Arg::I(hd as i32), Arg::I(c.rope_dim as i32), Arg::F(c.rope_base), Arg::P(self.rope_freqs.as_ref().map_or(0, |b| b.ptr)), Arg::I(c.rope_neox as i32), Arg::F(c.norm_eps)],
                )
            })?;
            // wide heads: prompt batches through attention_wide; one-token steps and verify batches (the same
            // per token) through the split GQA kernel, which reads each kv row once for all its query heads
            let split = self.gqa_kernel() && hd == 256 && (n < TILED_MIN_BATCH || s.exact.get()) && !old_wide_attention();
            if hd > ATT_MAX_HD && !split {
                let scale = 1.0 / (hd as f32).sqrt();
                let g = c.n_heads / c.n_kv_heads;
                // prompt batches: the tensor cores (f16 inputs, f32 sums: not attention_wide's bits); else groups
                // of WIDE_G query heads sharing their kv head's reads (attention_wide's bits); else attention_wide
                let tc = k.attention_prefill_tc256.filter(|_| hd == 256 && !old_wide_attention());
                let mode = if tc.is_some() { 2 } else if g % WIDE_G == 0 && !old_wide_attention() { 1 } else { 0 };
                let wide = |out: u64, mode: u8| {
                    if let (2, Some(f)) = (mode, tc) {
                        let args = [Arg::P(s.qkv.ptr), Arg::P(kc), Arg::P(vc), Arg::P(ks), Arg::P(vs), Arg::P(out), Arg::I(s.pos as i32), Arg::I(n as i32), Arg::I(c.n_heads as i32), Arg::I(c.n_kv_heads as i32), Arg::I(hd as i32), Arg::F(scale)];
                        return k.launch3(f, (c.n_heads as u32, n.div_ceil(PREFILL_TC_QT) as u32, 2), 128, 0, &args);
                    }
                    let args = [Arg::P(s.qkv.ptr), Arg::P(kc), Arg::P(vc), Arg::P(ks), Arg::P(vs), Arg::P(out), Arg::I(s.pos as i32), Arg::I(c.n_heads as i32), Arg::I(c.n_kv_heads as i32), Arg::I(hd as i32), Arg::F(scale)];
                    if mode == 1 {
                        k.launch3(k.attention_wide_g, (n as u32, c.n_kv_heads as u32, (g / WIDE_G) as u32), 256, 0, &args)
                    } else {
                        k.launch(k.attention_wide, (n as u32, c.n_heads as u32), 256, 0, &args)
                    }
                };
                self.timed(4, || wide(s.att.ptr, mode))?;
                if mode > 0 && check_wide_attention() {
                    // GHOST_INFER_CHECK_WIDE=1: compare with attention_wide on the same inputs (bits, and the
                    // largest difference relative to the largest value)
                    let len = n * c.n_heads * hd;
                    let other = DeviceBuf::alloc(len * 4)?;
                    wide(other.ptr, 0)?;
                    let (mut x, mut y) = (vec![0f32; len], vec![0f32; len]);
                    k.d2h(&mut x, s.att.ptr)?;
                    k.d2h(&mut y, other.ptr)?;
                    let diff = x.iter().zip(&y).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    let worst = x.iter().zip(&y).fold(0f32, |w, (a, b)| w.max((a - b).abs()));
                    let big = y.iter().fold(0f32, |w, b| w.max(b.abs()));
                    static CALLS: AtomicUsize = AtomicUsize::new(0);
                    static DIFFS: AtomicUsize = AtomicUsize::new(0);
                    let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
                    let diffs = DIFFS.fetch_add(diff, Ordering::Relaxed) + diff;
                    if (mode == 1 && diff > 0) || calls % 10 == 0 {
                        eprintln!("[wide attention check: {calls} calls, {diffs} values differ (this call {diff} of {len}, largest difference {worst:e} of values up to {big:e}, {n} tokens at {})]", s.pos);
                    }
                }
            } else {
                self.timed(4, || self.attention(s, n, (kc, vc, ks, vs), posp, fixed_split))?;
            }
            if c.attn_gated {
                let len = n * c.n_heads * hd;
                self.timed(4, || k.launch(k.sigmoid_mul, (len.div_ceil(256) as u32, 1), 256, 0, &[Arg::P(s.att.ptr), Arg::P(s.h.ptr), Arg::I(len as i32)]))?;
            }
            self.timed(5, || self.matmul_one(&layer.wo, s.att.ptr, s.x.ptr, n, true, s))?;

            self.timed(6, || self.ffn(l, layer, n, s))?;
        }

        if want != Out::None {
            self.timed(7, || {
                // Last: the last token's logits; All: every token's; Argmax: every token's, reduced on the GPU;
                // Hidden: the last token's normed state only (left in s.xb)
                let rows = if matches!(want, Out::Last | Out::Hidden) { 1 } else { n };
                let first = s.x.at((n - rows) * dim * 4);
                k.launch(k.rmsnorm, (rows as u32, 1), 256, 0, &[Arg::P(first), Arg::P(self.output_norm.ptr), Arg::P(s.xb.ptr), Arg::I(dim as i32), Arg::F(c.norm_eps)])?;
                if want == Out::Hidden {
                    return Ok(());
                }
                self.matmul_one(self.output_matrix(), s.xb.ptr, s.logits_dev.ptr, rows, false, s)?;
                if want == Out::Argmax {
                    k.launch(k.argmax_rows, (rows as u32, 1), 1024, 0, &[Arg::P(s.logits_dev.ptr), Arg::I(c.vocab as i32), Arg::P(s.picks.ptr)])?;
                }
                Ok(())
            })?;
        }
        Ok(())
    }
}

/// What a forward pass hands back.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Out {
    None,
    /// the last token's logits, copied to `GpuSession::logits`
    Last,
    /// the greedy next token after every token of the batch
    Argmax,
    /// every token's logits, copied to `GpuSession::rows`
    All,
    /// the last token's final-normed state, left on the GPU (GpuModel::hidden, head_logits)
    Hidden,
}

/// What a forward pass reads: token ids, or rows of `dim` floats (Embd::Vectors).
#[derive(Clone, Copy)]
enum Input<'a> {
    Tokens(&'a [u32]),
    Vectors(&'a [f32]),
}

/// A conversation the GPU is not holding: everything it left in the caches, kept in host memory, so that a
/// request which continues it can take the caches back without the model reading its history a second time
/// (`GpuSession::park`, `GpuSession::restore`).
///
/// The card holds one context at a time — the caches are a single sequence, indexed by position — so a
/// second conversation can only have them if the first one's contents are moved out. They go to host memory
/// rather than being thrown away because the first one is not finished: its client will send it again on the
/// next turn, and reading a long conversation back costs tens of seconds while moving it costs milliseconds.
pub struct Parked {
    /// The tokens the caches held, in order: what a later request is matched against to tell whether it
    /// continues this conversation.
    pub tokens: Vec<u32>,
    /// Where it stopped: the caches held `0..pos`, and this is what a request continuing it must be given
    /// room for.
    pub pos: usize,
    /// When it was last taken out of the caches, so that the least recently used one is the one dropped
    /// when host memory runs short.
    pub used: u64,
    /// Each cache's layers in turn, each holding that layer's first `pos` rows.
    parts: Vec<Vec<u8>>,
    /// The Gated DeltaNet layers' convolution and recurrent states. The one part with no position of its
    /// own, so unlike the rows above it cannot be recovered from the cache at any other point.
    delta: Vec<u8>,
    /// The states this conversation could come back to, as `(position, state)`. A request that continues it
    /// usually stops a few tokens short of where it got — the previous reply's tail is not in the new prompt
    /// — and the recurrent layers can only rewind to a note like one of these, never to an arbitrary
    /// position. Without them, coming back means reading from the start.
    snaps: Vec<(usize, Vec<u8>)>,
    /// The note of the head its session's conversations share. It travels with the conversation, so that a
    /// request which shares only that head still finds a note for it after this one has been put back.
    head: Option<(usize, Vec<u8>)>,
}

impl Parked {
    /// Bytes of host memory this holds, which is what `GHOST_INFER_PARK_MB` bounds. It comes to about the
    /// size of the conversation itself: the caches are one row per position per layer, so a copy of them is
    /// the same bytes the card was holding, plus the states, which are a fixed size.
    pub fn bytes(&self) -> usize {
        self.parts.iter().map(Vec::len).sum::<usize>()
            + self.delta.len()
            + self.snaps.iter().map(|(_, s)| s.len()).sum::<usize>()
            + self.head.iter().map(|(_, s)| s.len()).sum::<usize>()
    }

}

pub struct GpuSession {
    pub pos: usize,
    pub ctx: usize, // the most positions it may hold
    cap: usize,     // the positions its caches hold now (ctx, unless it grows: session_grows)
    grows: bool,
    tokens: DeviceBuf,
    x: DeviceBuf,
    xb: DeviceBuf,
    qkv: DeviceBuf,
    att: DeviceBuf,
    gu: DeviceBuf,
    h: DeviceBuf,
    k_cache: LayerCache,
    v_cache: LayerCache,
    k_scale: LayerCache, // int8 cache: f32 scale per (layer, position, kv head)
    v_scale: LayerCache,
    xq: DeviceBuf,      // int8 activations for the tiled matmul
    xd: DeviceBuf,       // their per-32 scales
    xsum: DeviceBuf,     // and per-32 sums of the int8 values
    att_part: DeviceBuf, // flash-decoding partial results
    logits_dev: DeviceBuf, // up to VERIFY_MAX_BATCH rows
    picks: DeviceBuf,      // their argmax
    pos_dev: DeviceBuf,    // the position, for CUDA graphs
    moe: std::cell::RefCell<MoeHost>, // host scratch of the CPU expert path
    moe_tab: DeviceBuf,               // a token's GPU-resident experts: matrix addresses, then weights
    graphs: [Ptr; 3],      // captured one-token steps: Last, Argmax, Hidden
    att_scr: DeviceBuf,    // the tiled verify attention's scores and weights (vfy_attention_fits models)
    exact: std::cell::Cell<bool>, // this pass checks a draft: verify kernels, no prompt kernels
    /// Gated DeltaNet layers: every layer's convolution state, then every layer's recurrent state (f32).
    /// This is the live state, the one `pos` stands at; notes of it at earlier positions are in `snaps`.
    delta: Option<DeviceBuf>,
    /// Notes taken at earlier positions: the recurrent state as it was, so a request that follows this
    /// conversation less far than it got can go back to one of them. They live in **host** memory, for the
    /// same reason a parked conversation does: a note is for a position this session has already moved past,
    /// so nothing is reading it now, and unlike the live state below it is not read again on every token —
    /// only when a request comes back to that point. Copies on the card would take memory away from the
    /// expert cache for nothing, while moving one back costs a few milliseconds, once.
    snaps: Vec<(usize, Vec<u8>)>,
    /// The note of the head that every conversation of a session shares: everything up to the end of its
    /// system block. It is held apart from the ring above because the ring drops the oldest note, and this
    /// head *is* the oldest — it is written first, in every prompt. Yet it is also the one part every new
    /// conversation of the session pays for again (a subagent shares nothing else), so it is the last note
    /// worth dropping rather than the first.
    head: Option<(usize, Vec<u8>)>,
    pub logits: Vec<f32>,
    pub rows: Vec<f32>, // Out::All: a batch's logits, one vocab-long row per token
}

impl GpuModel {
    /// The most tokens a verify batch (forward_argmax, forward_logits) may hold: VERIFY_MAX_BATCH on the
    /// tensor-core verify kernels, when every matrix is Q4_K or Q6_K and attention is not latent; else
    /// DECODE_MAX_BATCH.
    pub fn verify_max(&self) -> usize {
        let kq = |m: &GpuMatrix| matches!(m.fmt, Fmt::Q4K | Fmt::Q6K);
        let layers = self.layers.iter().all(|l| {
            l.mla.is_none()
                && l.qkv.iter().all(|(m, _)| kq(m))
                && kq(&l.wo)
                && matches!(&l.ffn, GpuFfn::Dense { gate_up, down } if gate_up.iter().all(|(m, _)| kq(m)) && kq(down))
        });
        if self.k.matmul_vfy.is_some() && !crate::quant::exact_mode() && layers && kq(self.output_matrix()) {
            VERIFY_MAX_BATCH
        } else {
            DECODE_MAX_BATCH
        }
    }

    /// The GPU expert cache's state, if this model has one yet.
    pub fn expert_cache(&self) -> Option<String> {
        self.cache.borrow().as_ref().map(ExpertCache::describe)
    }

    /// Grows the session's caches until they hold `need` positions, so a parked conversation can go back in
    /// (`GpuSession::restore`). The memory comes from the expert cache, the same as for a long prompt: the
    /// two trade against each other, and a conversation that is coming back needs the positions it stood at
    /// more than the cache needs the slots. A conversation parked from a session that already held this much
    /// costs nothing here — it is the same session, so the caches never shrank below it.
    pub fn reserve(&self, s: &mut GpuSession, need: usize) -> Result<(), String> {
        self.fit_session(s, need.min(s.ctx), true)
    }

    /// A session that grows gets room for `need` positions, in CTX_STEP steps: the memory it takes comes
    /// from the expert cache's last blocks, the memory it gives back goes to the cache. `shrink`: also give
    /// memory back, when the conversation needs two steps less than the session holds (a new one after a
    /// long one; not for every prompt that is a little shorter than the last).
    fn fit_session(&self, s: &mut GpuSession, need: usize, shrink: bool) -> Result<(), String> {
        if !s.grows {
            return Ok(());
        }
        let cap = (need.div_ceil(CTX_STEP).max(1) * CTX_STEP).min(s.ctx);
        if cap == s.cap || (cap < s.cap && !(shrink && cap + 2 * CTX_STEP <= s.cap)) {
            return Ok(());
        }
        let a = api()?;
        a.check(unsafe { (a.f.cuCtxSynchronize)() }, "synchronize")?; // no kernel queued still reads the old caches
        let c = &self.cfg;
        let mut cache = self.cache.borrow_mut();
        let sized = fixed_expert_cache().is_none();
        if let (Some(cache), Some(moe), true) = (cache.as_mut(), self.moe.as_ref(), sized) {
            let more = GpuSession::bytes(c, cap).saturating_sub(GpuSession::bytes(c, s.cap));
            let short = (more + GpuSession::resize_peak(c, cap) + reserve()).saturating_sub(free_memory()?);
            if short > 0 {
                cache.free_blocks(moe, short);
            }
        }
        let from = s.cap;
        s.resize(c, cap)?;
        if let (Some(cache), true) = (cache.as_mut(), sized) {
            cache.add_blocks(free_memory()?.saturating_sub(reserve()));
        }
        let slots = cache.as_ref().map_or(String::new(), |c| format!(", expert cache {} slots", c.slots()));
        eprintln!("[context memory {from} -> {cap} tokens{slots}]");
        Ok(())
    }
}

impl Drop for GpuSession {
    fn drop(&mut self) {
        self.drop_graphs();
    }
}

/// A cache of every position, per layer: a buffer of `cap` rows of `row` bytes for each layer (none when
/// the model keeps no such cache), so that it can grow or shrink a layer at a time.
struct LayerCache {
    row: usize,
    layers: Vec<DeviceBuf>,
}

impl LayerCache {
    fn new(row: usize, n_layers: usize, cap: usize) -> Result<LayerCache, String> {
        let layers = if row == 0 { Vec::new() } else { (0..n_layers).map(|_| DeviceBuf::alloc(cap * row)).collect::<Result<_, _>>()? };
        Ok(LayerCache { row, layers })
    }

    /// Layer l's first row (0 when there is no cache).
    fn at(&self, l: usize) -> u64 {
        self.layers.get(l).map_or(0, |b| b.ptr)
    }

    /// To `cap` rows a layer, keeping the first `keep`.
    fn resize(&mut self, cap: usize, keep: usize) -> Result<(), String> {
        let a = api()?;
        for b in &mut self.layers {
            let fresh = DeviceBuf::alloc(cap * self.row)?;
            if keep > 0 {
                a.check(unsafe { (a.f.cuMemcpyDtoD_v2)(fresh.ptr, b.ptr, keep * self.row) }, "context cache copy")?;
                // the copy does not wait for the host: it is done before the old buffer is freed
                a.check(unsafe { (a.f.cuCtxSynchronize)() }, "synchronize")?;
            }
            *b = fresh;
        }
        Ok(())
    }
}

/// One-token steps replay CUDA graphs unless GHOST_INFER_NO_GRAPH is set or the profiler (which
/// synchronises after every operation) is on.
fn graphs_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GHOST_INFER_NO_GRAPH").is_none()) && !profile::enabled()
}

impl GpuSession {
    pub fn new(cfg: &Config, ctx: usize) -> Result<GpuSession, String> {
        let ctx = ctx.min(cfg.max_ctx);
        let grows = session_grows(cfg);
        let cap = if grows { ctx.min(CTX_STEP) } else { ctx };
        let [xq, xd, xsum, att_part, tokens, x, xb, qkv, att, gu, h, logits_dev, picks, pos_dev, att_scr] = Self::sizes(cfg, cap).map(DeviceBuf::alloc);
        let [k_row, v_row, ks_row, vs_row] = Self::cache_rows(cfg);
        Ok(GpuSession {
            xq: xq?,
            xd: xd?,
            xsum: xsum?,
            att_part: att_part?,
            pos: 0,
            ctx,
            cap,
            grows,
            tokens: tokens?,
            x: x?,
            xb: xb?,
            qkv: qkv?,
            att: att?,
            gu: gu?,
            h: h?,
            k_cache: LayerCache::new(k_row, cfg.n_attn_layers(), cap)?,
            v_cache: LayerCache::new(v_row, cfg.n_attn_layers(), cap)?,
            k_scale: LayerCache::new(ks_row, cfg.n_attn_layers(), cap)?,
            v_scale: LayerCache::new(vs_row, cfg.n_attn_layers(), cap)?,
            delta: match Self::delta_bytes(cfg) {
                0 => None,
                bytes => {
                    let buf = DeviceBuf::alloc(bytes)?;
                    zero(&buf)?;
                    Some(buf)
                }
            },
            snaps: Vec::new(),
            head: None,
            logits_dev: logits_dev?,
            picks: picks?,
            pos_dev: pos_dev?,
            moe: Default::default(),
            moe_tab: DeviceBuf::alloc(4096)?,
            graphs: [std::ptr::null_mut(); 3],
            att_scr: att_scr?,
            exact: Default::default(),
            logits: vec![0.0; cfg.vocab],
            rows: Vec::new(),
        })
    }

    /// To caches of `cap` positions, keeping the first `pos`.
    fn resize(&mut self, cfg: &Config, cap: usize) -> Result<(), String> {
        let keep = self.pos.min(cap);
        for c in [&mut self.k_cache, &mut self.v_cache, &mut self.k_scale, &mut self.v_scale] {
            c.resize(cap, keep)?;
        }
        let part = Self::sizes(cfg, cap)[3];
        if part != self.att_part.bytes {
            self.att_part = DeviceBuf::alloc(part)?;
        }
        self.drop_graphs(); // they hold the old addresses
        self.cap = cap;
        Ok(())
    }

    fn drop_graphs(&mut self) {
        if let Ok(a) = api() {
            for g in self.graphs.iter_mut().filter(|g| !g.is_null()) {
                unsafe { (a.f.cuGraphExecDestroy)(*g) };
                *g = std::ptr::null_mut();
            }
        }
    }

    /// Device memory of a session holding `ctx` positions (with room for the DeltaNet snapshots).
    fn bytes(cfg: &Config, ctx: usize) -> usize {
        Self::sizes(cfg, ctx).iter().sum::<usize>() + cfg.n_attn_layers() * ctx * Self::cache_rows(cfg).iter().sum::<usize>() + Self::delta_bytes(cfg)
    }

    /// Bytes of every DeltaNet layer's states (0 for other models).
    fn delta_bytes(cfg: &Config) -> usize {
        cfg.delta.map_or(0, |d| 4 * (cfg.n_layers - cfg.n_attn_layers()) * (d.conv_len() + d.state_len()))
    }

    /// DeltaNet layer `slot`'s convolution state and recurrent state.
    fn delta_state(&self, cfg: &Config, slot: usize) -> (u64, u64) {
        let (d, buf) = (cfg.delta.unwrap(), self.delta.as_ref().unwrap());
        let layers = cfg.n_layers - cfg.n_attn_layers();
        (buf.at(4 * slot * d.conv_len()), buf.at(4 * (layers * d.conv_len() + slot * d.state_len())))
    }

    /// A recurrent model (Gated DeltaNet) cannot cut its state back to an earlier position, only go back to a
    /// snapshot or to the start.
    pub fn recurrent(&self) -> bool {
        self.delta.is_some()
    }

    pub fn reset(&mut self) -> Result<(), String> {
        self.pos = 0;
        // The notes go with the history they describe: this session is holding none of it any more, so a
        // note could only send a later request back into a state that no longer matches its tokens. The
        // shared head stays — it belongs to the session rather than to this conversation, and it is exactly
        // the note a request that shares only the system block is looking for.
        self.snaps.clear();
        self.delta.as_ref().map_or(Ok(()), zero)
    }

    /// Copies the recurrent state out to host memory as a note to come back to.
    ///
    /// The caller takes one where a **reply** ends, and nowhere else. That is where a request comes back
    /// to: a retry, a rewind, a fork, a subagent cutting in. A prompt's end is not — after a prompt comes
    /// the reply head and then the reply, so the position the next request wants is the one at the end of
    /// what was generated, not the one before it.
    ///
    /// The ring keeps as many notes as `note_budget()` allows and drops the oldest. That decides which
    /// requests pay: what stays covers the positions this conversation passed through recently, which is
    /// where a fork, a rewind or a retry comes back to. A request whose prefix stops before the oldest note
    /// kept reads its history again — exactly what it would have done before any of this existed.
    pub fn snapshot(&mut self) -> Result<(), String> {
        if self.delta.as_ref().is_none_or(|b| b.bytes == 0) || self.snaps.iter().any(|s| s.0 == self.pos) {
            return Ok(());
        }
        let Some(copy) = self.note_out()? else { return Ok(()) };
        // the shared head holds the earliest position already, so the ring keeps the latest few
        if self.snaps.len() == MAX_SNAPS {
            self.snaps.remove(0);
        }
        self.snaps.push((self.pos, copy));
        Ok(())
    }

    /// Leaves the note of the shared head. It is kept outside the ring, so no budget can evict it: a second
    /// conversation of the same session — a subagent, or a new chat carrying the same system prompt — shares
    /// nothing with this one *except* that block, and with a note at its end it reads only what follows. The
    /// head is written first in every prompt, so a plain ring would always call it the oldest note and throw
    /// it away first, which is the opposite of what is wanted. Nothing bounds it: it is one note, and a
    /// session has one system block.
    pub fn snapshot_head(&mut self) -> Result<(), String> {
        let Some(copy) = self.note_out()? else { return Ok(()) };
        let at = self.pos;
        self.snaps.retain(|(p, _)| *p != at);
        self.head = Some((at, copy));
        Ok(())
    }

    /// The state at the current position, in host memory, or `None` for a model without recurrent layers.
    fn note_out(&self) -> Result<Option<Vec<u8>>, String> {
        let Some(buf) = self.delta.as_ref().filter(|b| b.bytes > 0) else { return Ok(None) };
        let a = api()?;
        let mut copy = vec![0u8; buf.bytes];
        a.check(unsafe { (a.f.cuMemcpyDtoH_v2)(copy.as_mut_ptr() as *mut c_void, buf.ptr, copy.len()) }, "DeltaNet note")?;
        Ok(Some(copy))
    }

    /// Goes back to at most `pos` positions and returns how many are kept.
    ///
    /// A model without recurrent layers keeps `pos` exactly: its caches are a sequence indexed by position,
    /// so dropping `pos` from the end costs nothing and needs no copy. A recurrent one cannot stand at an
    /// arbitrary position — its state carries no position of its own — so it can only go back to a note
    /// taken at or before `pos` (from the ring, or the shared `head`), or start over when there is none. The
    /// note comes back from host memory into the live state.
    pub fn truncate(&mut self, pos: usize) -> Result<usize, String> {
        if pos >= self.pos {
            return Ok(self.pos);
        }
        if !self.recurrent() {
            self.pos = pos;
            return Ok(pos);
        }
        self.snaps.retain(|s| s.0 <= pos);
        let at = self
            .snaps
            .iter()
            .map(|(at, _)| *at)
            .chain(self.head.iter().filter(|(at, _)| *at <= pos).map(|(at, _)| *at))
            .max();
        let Some(at) = at else {
            self.reset()?;
            return Ok(self.pos);
        };
        let note = self
            .snaps
            .iter()
            .find(|(p, _)| *p == at)
            .map(|(_, s)| s)
            .or_else(|| self.head.as_ref().filter(|(p, _)| *p == at).map(|(_, s)| s));
        if let (Some(copy), Some(buf)) = (note, self.delta.as_ref()) {
            if !copy.is_empty() {
                let a = api()?;
                a.check(unsafe { (a.f.cuMemcpyHtoD_v2)(buf.ptr, copy.as_ptr() as *const c_void, copy.len()) }, "DeltaNet restore")?;
            }
        }
        self.pos = at;
        Ok(self.pos)
    }

    /// Copies everything a conversation has written into the caches out to host memory, so another one can
    /// have the GPU.
    ///
    /// The whole context goes, not just the part a later request will not recompute. The caches are a single
    /// sequence at positions `0..pos` and the next reader writes from wherever it starts, so keeping any of
    /// it on the card is a gamble on nobody having written there. The copy is the conversation's tokens, the
    /// four caches' first `pos` rows, the DeltaNet state, and the notes that state can be rewound to. It
    /// comes to about the size of the conversation, which is why what bounds how many are kept is host
    /// memory rather than the card.
    pub fn park(&mut self, tokens: Vec<u32>) -> Result<Parked, String> {
        let a = api()?;
        let pos = self.pos.min(self.cap);
        let mut parts = Vec::new();
        for c in [&self.k_cache, &self.v_cache, &self.k_scale, &self.v_scale] {
            for b in &c.layers {
                let mut buf = vec![0u8; pos * c.row];
                if !buf.is_empty() {
                    a.check(unsafe { (a.f.cuMemcpyDtoH_v2)(buf.as_mut_ptr() as *mut c_void, b.ptr, buf.len()) }, "park the attention cache")?;
                }
                parts.push(buf);
            }
        }
        let mut delta = vec![0u8; self.delta.as_ref().map_or(0, |b| b.bytes)];
        if let Some(b) = self.delta.as_ref().filter(|_| !delta.is_empty()) {
            a.check(unsafe { (a.f.cuMemcpyDtoH_v2)(delta.as_mut_ptr() as *mut c_void, b.ptr, delta.len()) }, "park the DeltaNet state")?;
        }
        // The notes are host memory already, so they travel as they are — no round trip. The state at `pos`
        // is `delta` itself, so a note there would only repeat what was read out a moment ago; the ones
        // below it are the ones a later request can actually use.
        let snaps = self.snaps.iter().filter(|(at, _)| *at < pos).cloned().collect();
        Ok(Parked { tokens, pos, parts, delta, snaps, head: self.head.clone(), used: 0 })
    }

    /// Puts a parked conversation back into the caches, which must already hold room for `parked.pos`
    /// positions (see `GpuModel::reserve`). Its token list is the caller's.
    ///
    /// The notes go back with it, and that is what makes this worth doing at all. A request that continues
    /// the conversation almost never follows it all the way — the previous reply's tail is not part of the
    /// new prompt — so the session is left a few tokens short of `pos` and has to step back to a note at or
    /// before that point. Restoring without the notes leaves it with none, and a recurrent model with no
    /// note to step back to has to read its history from the beginning: the caches came back, and then
    /// nothing used them.
    pub fn restore(&mut self, parked: &Parked) -> Result<(), String> {
        let a = api()?;
        if parked.pos > self.cap {
            return Err(format!("the caches hold {} positions, the parked context has {}", self.cap, parked.pos));
        }
        let mut i = 0usize;
        for c in [&self.k_cache, &self.v_cache, &self.k_scale, &self.v_scale] {
            let _ = c.row;
            for b in &c.layers {
                if let Some(part) = parked.parts.get(i).filter(|p| !p.is_empty()) {
                    a.check(unsafe { (a.f.cuMemcpyHtoD_v2)(b.ptr, part.as_ptr() as *const c_void, part.len()) }, "put back a parked attention cache")?;
                }
                i += 1;
            }
        }
        if let Some(b) = self.delta.as_ref().filter(|_| !parked.delta.is_empty()) {
            a.check(unsafe { (a.f.cuMemcpyHtoD_v2)(b.ptr, parked.delta.as_ptr() as *const c_void, parked.delta.len()) }, "put back a parked DeltaNet state")?;
        }
        // the notes come back as they were, so the step back that follows has somewhere to land
        self.snaps = parked.snaps.clone();
        self.head = parked.head.clone();
        // the state just restored is where this conversation now stands: leave a note here too, so the next
        // request that continues it has one at the position this one reached
        self.pos = parked.pos;
        self.snapshot()?;
        Ok(())
    }

    /// The most memory `resize` to `ctx` positions holds besides the old caches and the new ones done so
    /// far: one new buffer.
    fn resize_peak(cfg: &Config, ctx: usize) -> usize {
        Self::cache_rows(cfg).iter().map(|r| r * ctx).max().unwrap_or(0).max(Self::sizes(cfg, ctx)[3])
    }

    /// Bytes per position and layer of the caches: keys, values, and the int8 cache's key and value scales.
    /// Latent attention keeps one f16 (kl + rd)-wide row (no separate values, no int8 option).
    fn cache_rows(cfg: &Config) -> [usize; 4] {
        if cfg.mla.is_some() {
            return [2 * cfg.k_row(), 0, 0, 0];
        }
        let kv = cfg.n_kv_heads * cfg.head_dim * kv_bytes();
        let scales = if kv_int8() { 4 * cfg.n_kv_heads } else { 0 };
        [kv, kv, scales, scales]
    }

    /// Bytes of each device buffer besides the caches, in the order `new` takes them.
    fn sizes(cfg: &Config, ctx: usize) -> [usize; 15] {
        if let Some(m) = cfg.mla {
            return Self::sizes_mla(cfg, m);
        }
        let (q_dim, kv_dim) = (cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim);
        // qwen3next: DeltaNet's [q k v | z | ba] rows and gated attention's [q | gate] ones go through s.gu, its
        // outputs (value heads) through s.att, its gates and the attention gates through s.h
        let d = cfg.delta.unwrap_or(crate::model::Delta { k_heads: 0, v_heads: 0, head: 0, conv: 0, every: 1, tiled: false });
        let gu = (2 * cfg.ffn_dim).max(d.conv_dim() + d.value_dim() + 2 * d.v_heads).max(if cfg.attn_gated { 2 * q_dim + 2 * kv_dim } else { 0 });
        let att = q_dim.max(d.value_dim());
        let widest = cfg.dim.max(att).max(cfg.ffn_dim);
        // split attention runs at most SPLIT_MAX_BATCH tokens at a time, with slices of >= 32 positions
        let slices = SPLIT_MAX_BATCH * ctx.div_ceil(32);
        [
            max_batch(cfg) * widest,
            4 * max_batch(cfg) * widest / 32,
            4 * max_batch(cfg) * widest / 32,
            // wide heads: only one-token steps and verify batches split (prompt batches run attention_wide),
            // so their partial results are sized for the slices a token actually uses
            if cfg.head_dim > ATT_MAX_HD { 4 * SPLIT_MAX_BATCH * max_slices(cfg, ctx) * cfg.n_heads * (cfg.head_dim + 2) } else { 4 * slices * cfg.n_heads * (cfg.head_dim + 2) },
            4 * max_batch(cfg),
            4 * max_batch(cfg) * cfg.dim,
            4 * max_batch(cfg) * cfg.dim,
            4 * max_batch(cfg) * (q_dim + 2 * kv_dim),
            4 * max_batch(cfg) * att,
            4 * max_batch(cfg) * gu,
            4 * max_batch(cfg) * cfg.ffn_dim.max(q_dim).max(2 * d.v_heads),
            4 * VERIFY_MAX_BATCH * cfg.vocab,
            4 * VERIFY_MAX_BATCH,
            4,
            if vfy_attention_fits(cfg) { 4 * AV_SCR_ROWS * cfg.n_heads * ATT_MAX_CHUNK } else { 4 },
        ]
    }

    /// `sizes` for latent attention: s.qkv holds the queries then [low-rank query | compressed kv] rows;
    /// s.att a token's heads x kl outputs; s.gu also the queries moved into the latent space.
    fn sizes_mla(cfg: &Config, m: crate::model::Mla) -> [usize; 15] {
        let (h, kd) = (cfg.n_heads, cfg.k_row());
        let q_dim = h * cfg.head_dim;
        let att = (h * m.kv_lora).max(q_dim);
        let gu = (2 * cfg.ffn_dim.max(cfg.expert_ffn_dim * cfg.n_shared_experts)).max(h * kd);
        let widest = cfg.dim.max(q_dim).max(cfg.ffn_dim).max(att);
        [
            max_batch(cfg) * widest,
            4 * max_batch(cfg) * widest / 32,
            4 * max_batch(cfg) * widest / 32,
            4 * MLA_SPLIT_MAX_BATCH * MLA_MAX_SPLIT * h * (m.kv_lora + 2),
            4 * max_batch(cfg),
            4 * max_batch(cfg) * cfg.dim,
            4 * max_batch(cfg) * cfg.dim,
            4 * max_batch(cfg) * (q_dim + m.q_lora + kd),
            4 * max_batch(cfg) * att,
            4 * max_batch(cfg) * gu,
            4 * max_batch(cfg) * cfg.ffn_dim,
            4 * DECODE_MAX_BATCH * cfg.vocab,
            4 * DECODE_MAX_BATCH,
            4,
            4, // no tiled verify attention for latent attention
        ]
    }
}

/// GPU memory kept free when the context is sized automatically: the desktop, other programs, and
/// the driver's own allocations (CUDA graphs, launch buffers). On Windows going over does not fail:
/// the driver quietly moves memory to the much slower system RAM.
const FIT_RESERVE: usize = 384 << 20;

/// More kept free beside FIT_RESERVE (set while the voice is loaded: a call's app window, a browser or a screen
/// recorder opened meanwhile must fit without pushing the call's memory out to system RAM, where its speech
/// ran about seven times slower and stuttered).
static EXTRA_RESERVE: AtomicUsize = AtomicUsize::new(0);

pub fn set_extra_reserve(bytes: usize) {
    EXTRA_RESERVE.store(bytes, Ordering::Relaxed);
}

/// The GPU memory kept free: FIT_RESERVE and any extra.
fn reserve() -> usize {
    FIT_RESERVE + EXTRA_RESERVE.load(Ordering::Relaxed)
}

/// GPU memory for the expert cache: what is free when it is made (after the weights and the session),
/// less the reserve. GHOST_INFER_EXPERT_CACHE_MB sets it instead (0: every expert on the CPU).
fn expert_cache_budget() -> Result<usize, String> {
    match fixed_expert_cache() {
        Some(bytes) => Ok(bytes),
        None => Ok(free_memory()?.saturating_sub(reserve())),
    }
}

/// GHOST_INFER_EXPERT_CACHE_MB's size, in bytes.
fn fixed_expert_cache() -> Option<usize> {
    std::env::var("GHOST_INFER_EXPERT_CACHE_MB").ok().and_then(|v| v.parse::<usize>().ok()).map(|mb| mb << 20)
}

fn free_memory() -> Result<usize, String> {
    let a = api()?;
    let (mut free, mut _total) = (0usize, 0usize);
    a.check(unsafe { (a.f.cuMemGetInfo_v2)(&mut free, &mut _total) }, "memory info")?;
    Ok(free)
}

/// For deciding what fits before loading it: the GPU memory free now, what stays free beside the models, and
/// what a session of `ctx` positions takes.
pub fn gpu_free() -> Result<usize, String> {
    free_memory()
}

pub fn kept_free() -> usize {
    reserve()
}

pub fn session_bytes(cfg: &Config, ctx: usize) -> usize {
    GpuSession::bytes(cfg, ctx)
}

/// Experts' slots per block of GPU memory (the expert cache grows and shrinks a block at a time).
const SLOT_BLOCK: usize = 16;

/// Copies of the DeltaNet states a session keeps besides the shared head: one per reply that ended here,
/// the oldest making way. They are in host memory, so the count is all there is to bound.
const MAX_SNAPS: usize = 3;

/// A layer's experts in one 2-bit format (gate, up and down alike): its `fmt` for the IQ2 kernels.
fn iq2_fmt(t: &[ExpertTensor; 3]) -> Option<i32> {
    let fmt = match t[0].ty {
        GgmlType::Iq2Xxs => 0,
        GgmlType::Iq2S => 1,
        _ => return None,
    };
    t.iter().all(|m| m.ty == t[0].ty && m.cols % 256 == 0).then_some(fmt)
}

/// Zeroes a device buffer (from the host: the driver API loaded here has no memset).
fn zero(buf: &DeviceBuf) -> Result<(), String> {
    buf.write(&vec![0u8; buf.bytes])
}

/// GHOST_INFER_OLD_WIDE_ATT=1: one-token steps of 256-wide heads through attention_wide as before (for comparisons).
fn old_wide_attention() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GHOST_INFER_OLD_WIDE_ATT").is_some())
}

/// GHOST_INFER_CHECK_WIDE=1: prompt batches of 256-wide heads also run attention_wide and compare its bits.
fn check_wide_attention() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GHOST_INFER_CHECK_WIDE").is_some())
}

/// Query heads attention_wide_g takes per block (must match WIDE_G).
const WIDE_G: usize = 4;

/// Most slices a token of a ctx-position session splits its attention into (GpuModel::max_splits, from the
/// config alone).
fn max_slices(cfg: &Config, ctx: usize) -> usize {
    let gqa = cfg.head_dim % 32 == 0 && cfg.n_heads % cfg.n_kv_heads == 0 && cfg.n_heads / cfg.n_kv_heads <= ATT_MAX_G;
    let want = ATT_TARGET_BLOCKS.div_ceil(if gqa { cfg.n_kv_heads } else { cfg.n_heads });
    (1..=ctx).map(|span| span.div_ceil(GpuModel::att_chunk(span, want))).max().unwrap_or(1)
}

/// A mixture of experts' session holds its positions in steps of this many: the memory of positions a
/// conversation has not reached yet is expert cache meanwhile.
const CTX_STEP: usize = 4096;

/// Sessions that grow with the conversation: a mixture of experts' (on the GPU, its expert cache wants every
/// byte), unless GHOST_INFER_FULL_CTX is set.
pub fn session_grows(cfg: &Config) -> bool {
    static FULL: OnceLock<bool> = OnceLock::new();
    cfg.is_moe() && !*FULL.get_or_init(|| std::env::var_os("GHOST_INFER_FULL_CTX").is_some())
}

/// Prompt batches of at least this many tokens run every expert on the GPU (smaller ones on the CPU).
const MOE_GPU_MIN_BATCH: usize = 32;

/// GHOST_INFER_MOE_STATS=1: where a mixture of experts' one-token steps go. Per layer step: picks found in the
/// expert cache and not; time waiting for the GPU (the copy to the host), routing, the CPU's experts, the copy
/// back, and the cache's bookkeeping; uploads queued.
static MOE_STATS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
const MS_UPLOADS: usize = 8;
const MS_HITS: usize = 0;
const MS_MISSES: usize = 1;
const MS_STEPS: usize = 2;
const MS_WAIT: usize = 3;
const MS_ROUTE: usize = 4;
const MS_CPU: usize = 5;
const MS_BACK: usize = 6;
const MS_NOTE: usize = 7;
fn moe_stats_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GHOST_INFER_MOE_STATS").is_some())
}
fn moe_stat(i: usize, v: u64) {
    MOE_STATS[i].fetch_add(v, Ordering::Relaxed);
}
/// The one-token MoE steps since the last call, per token of the `tokens` generated; None when not collected.
pub fn moe_stats_take(tokens: usize) -> Option<String> {
    if !moe_stats_on() {
        return None;
    }
    let v: Vec<u64> = MOE_STATS.iter().map(|a| a.swap(0, Ordering::Relaxed)).collect();
    let tokens = tokens.max(1) as f64;
    let ms = |i: usize| v[i] as f64 / 1e6 / tokens;
    let hits = v[MS_HITS] as f64 / (v[MS_HITS] + v[MS_MISSES]).max(1) as f64;
    // page faults of the whole process (the upload thread's included) since the last report
    static FAULTS: AtomicU64 = AtomicU64::new(0);
    let (faults, working_set) = os::page_faults();
    let new_faults = faults.saturating_sub(FAULTS.swap(faults, Ordering::Relaxed));
    Some(format!(
        "[moe: {} layer steps, {:.1}% of expert picks in the GPU cache, {:.1} uploads and {:.0} page faults per token (working set {} MB); per token: waiting for the GPU {:.2} ms, routing {:.2} ms, CPU experts {:.2} ms, copy back {:.2} ms, cache upkeep {:.2} ms]",
        v[MS_STEPS],
        100.0 * hits,
        v[MS_UPLOADS] as f64 / tokens,
        new_faults as f64 / tokens,
        working_set >> 20,
        ms(MS_WAIT),
        ms(MS_ROUTE),
        ms(MS_CPU),
        ms(MS_BACK),
        ms(MS_NOTE)
    ))
}

/// Default context of a mixture-of-experts model on the GPU: the memory a longer one would take is
/// worth more as expert cache.
pub const MOE_DEFAULT_CTX: usize = 8192;

/// The longest context (a multiple of 1024, at most every model's limit) for which a session of each
/// of these models fits in the GPU memory now free.
pub fn fit_ctx(cfgs: &[&Config]) -> Result<usize, String> {
    let a = api()?;
    let (mut free, mut _total) = (0usize, 0usize);
    a.check(unsafe { (a.f.cuMemGetInfo_v2)(&mut free, &mut _total) }, "memory info")?;
    let need = |ctx: usize| cfgs.iter().map(|c| GpuSession::bytes(c, ctx)).sum::<usize>() + reserve();
    let limit = cfgs.iter().map(|c| c.max_ctx).min().unwrap_or(2048);
    let longest = || {
        let mut ctx = 1024.min(limit);
        while ctx + 1024 <= limit && need(ctx + 1024) <= free {
            ctx += 1024;
        }
        ctx
    };
    let mut ctx = longest();
    // a big model on a small card (Mistral Nemo 12B on 8 GB: 1024 positions): prompt batches of 1024 tokens keep
    // about 280 MB of activations, worth 3000 positions of int8 cache; smaller batches read the prompt a little slower
    let batch = || cfgs.iter().map(|c| max_batch(c)).max().unwrap_or(MAX_BATCH);
    // (not a mixture of experts: each prompt pass streams its experts, so more passes cost far more)
    while ctx < FIT_SHORT_CTX.min(limit) && batch() > MIN_FIT_BATCH && !cfgs.iter().any(|c| c.is_moe()) {
        BATCH_CAP.store(batch() / 2, Ordering::Relaxed);
        ctx = longest();
    }
    if BATCH_CAP.load(Ordering::Relaxed) != 0 {
        eprintln!("[note: prompts run {} tokens at a time to leave room for a {ctx}-token context]", batch());
    }
    if need(ctx) > free {
        eprintln!("[note: only {} MB of GPU memory is free; a {ctx}-token context will partly live in system RAM and run slowly]", free >> 20);
    }
    Ok(ctx)
}
