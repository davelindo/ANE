// model.h — Stories110M model struct + weight loading + ANE kernel compilation
// Training version: baked-weight conv kernels, recompile when weights update
#pragma once
#include "ane/core/ane_mil_gen.h"
#include "ane/core/ane_runtime.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_LAYERS 12
#define DIM 768
#define HIDDEN_DIM 2048
#define N_HEADS 12
#define HEAD_DIM 64
#define VOCAB_SIZE 32000
#define MAX_SEQ 1024

typedef struct {
  int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
} Config;

typedef struct {
  Config cfg;
  int seq_len; // training sequence length

  // Raw weights (f32)
  float *token_embedding;     // [vocab_size, dim]
  float *rms_att_w[N_LAYERS]; // [dim]
  float *wq[N_LAYERS];        // [dim, dim]
  float *wk[N_LAYERS];        // [dim, dim]
  float *wv[N_LAYERS];        // [dim, dim]
  float *wo[N_LAYERS];        // [dim, dim]
  float *rms_ffn_w[N_LAYERS]; // [dim]
  float *w1[N_LAYERS];        // [hidden_dim, dim]
  float *w2[N_LAYERS];        // [dim, hidden_dim]
  float *w3[N_LAYERS];        // [hidden_dim, dim]
  float *rms_final_w;         // [dim]
  float *wcls;                // [vocab_size, dim]

  // Per-layer ANE conv kernels (baked weights, recompiled on update)
  ANEKernel *kern_q[N_LAYERS];  // Q projection: dim→dim
  ANEKernel *kern_k[N_LAYERS];  // K projection: dim→dim
  ANEKernel *kern_v[N_LAYERS];  // V projection: dim→dim
  ANEKernel *kern_o[N_LAYERS];  // O projection: dim→dim
  ANEKernel *kern_w1[N_LAYERS]; // FFN w1: dim→hidden
  ANEKernel *kern_w2[N_LAYERS]; // FFN w2: hidden→dim
  ANEKernel *kern_w3[N_LAYERS]; // FFN w3: dim→hidden
  ANEKernel *kern_cls;          // Classifier: dim→vocab

  // Gradient accumulators (f32)
  float *grad_wq[N_LAYERS], *grad_wk[N_LAYERS], *grad_wv[N_LAYERS],
      *grad_wo[N_LAYERS];
  float *grad_w1[N_LAYERS], *grad_w2[N_LAYERS], *grad_w3[N_LAYERS];
  float *grad_wcls;
  float *grad_emb;

  // Adam optimizer state
  float *adam_m, *adam_v;
  int adam_step;
  size_t total_params;

  // Activation cache for backward
  float *act_x[N_LAYERS];
  float *act_xnorm[N_LAYERS];
  float *act_q[N_LAYERS];
  float *act_k[N_LAYERS];
  float *act_v[N_LAYERS];
  float *act_attn_out[N_LAYERS];
  float *act_ffn_in[N_LAYERS];
  float *act_h1[N_LAYERS];
  float *act_h3[N_LAYERS];
  float *act_silu[N_LAYERS];
  float *act_final;
  float *act_pre_final;
  float *logits;
} Model;

static int model_read_exact(FILE *f, void *ptr, size_t item_size, size_t count,
                            const char *path, const char *what) {
  if (count == 0)
    return 0;
  size_t got = fread(ptr, item_size, count, f);
  if (got == count)
    return 0;
  if (ferror(f)) {
    fprintf(stderr, "Read error (%s, %s): %s\n", path, what, strerror(errno));
  } else {
    fprintf(stderr, "Truncated model file (%s, %s): read %zu of %zu items\n",
            path, what, got, count);
  }
  return -1;
}

static int model_close_checked(FILE *f, const char *path) {
  if (fclose(f) == 0)
    return 0;
  fprintf(stderr, "Close error (%s): %s\n", path, strerror(errno));
  return -1;
}

static int model_mul_size(size_t a, size_t b, size_t *out) {
  if (a != 0 && b > SIZE_MAX / a)
    return -1;
  *out = a * b;
  return 0;
}

static int model_alloc_and_read(FILE *f, const char *path, const char *what,
                                float **out, size_t count) {
  *out = NULL;
  if (count == 0)
    return 0;
  *out = (float *)malloc(count * sizeof(float));
  if (!*out) {
    fprintf(stderr, "Out of memory allocating %s (%zu floats)\n", what, count);
    return -1;
  }
  if (model_read_exact(f, *out, sizeof(float), count, path, what) != 0) {
    free(*out);
    *out = NULL;
    return -1;
  }
  return 0;
}

static int model_load_weights(Model *m, const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Cannot open %s: %s\n", path, strerror(errno));
    return -1;
  }

  int rc = -1;
  Config cfg = {0};
  bool shared = false;
  int d = 0, hd = 0, nl = 0, vs = 0;
  size_t d_count = 0, hd_count = 0, nl_count = 0, vs_count = 0;
  size_t dd_count = 0, hdd_count = 0, dhd_count = 0;
  size_t token_count = 0;
  size_t nl_d_count = 0, nl_dd_count = 0, nl_hdd_count = 0, nl_dhd_count = 0;

  float *token_embedding = NULL;
  float *rms_att_all = NULL, *wq_all = NULL, *wk_all = NULL, *wv_all = NULL,
        *wo_all = NULL;
  float *rms_ffn_all = NULL, *w1_all = NULL, *w2_all = NULL, *w3_all = NULL;
  float *rms_att_w[N_LAYERS] = {0};
  float *wq[N_LAYERS] = {0}, *wk[N_LAYERS] = {0}, *wv[N_LAYERS] = {0},
        *wo[N_LAYERS] = {0};
  float *rms_ffn_w[N_LAYERS] = {0}, *w1[N_LAYERS] = {0}, *w2[N_LAYERS] = {0},
        *w3[N_LAYERS] = {0};
  float *rms_final_w = NULL;
  float *wcls = NULL;

  if (model_read_exact(f, &cfg, sizeof(Config), 1, path, "config") != 0)
    goto out;
  shared = cfg.vocab_size > 0;
  if (cfg.vocab_size < 0)
    cfg.vocab_size = -cfg.vocab_size;

  d = cfg.dim;
  hd = cfg.hidden_dim;
  nl = cfg.n_layers;
  vs = cfg.vocab_size;
  if (d <= 0 || hd <= 0 || nl != N_LAYERS || vs <= 0 || cfg.n_heads <= 0 ||
      cfg.seq_len <= 0) {
    fprintf(stderr,
            "Invalid model config in %s: dim=%d hidden=%d layers=%d heads=%d "
            "vocab=%d seq=%d\n",
            path, cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads,
            cfg.vocab_size, cfg.seq_len);
    goto out;
  }

  d_count = (size_t)d;
  hd_count = (size_t)hd;
  nl_count = (size_t)nl;
  vs_count = (size_t)vs;
  if (model_mul_size(d_count, d_count, &dd_count) != 0 ||
      model_mul_size(hd_count, d_count, &hdd_count) != 0 ||
      model_mul_size(d_count, hd_count, &dhd_count) != 0 ||
      model_mul_size(vs_count, d_count, &token_count) != 0 ||
      model_mul_size(nl_count, d_count, &nl_d_count) != 0 ||
      model_mul_size(nl_count, dd_count, &nl_dd_count) != 0 ||
      model_mul_size(nl_count, hdd_count, &nl_hdd_count) != 0 ||
      model_mul_size(nl_count, dhd_count, &nl_dhd_count) != 0) {
    fprintf(stderr, "Model dimensions overflow size calculations for %s\n",
            path);
    goto out;
  }

  printf("Model: dim=%d hidden=%d layers=%d heads=%d vocab=%d seq=%d\n",
         cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.vocab_size,
         cfg.seq_len);

  if (model_alloc_and_read(f, path, "token_embedding", &token_embedding,
                           token_count) != 0)
    goto out;
  if (model_alloc_and_read(f, path, "rms_att_all", &rms_att_all, nl_d_count) !=
      0)
    goto out;
  if (model_alloc_and_read(f, path, "wq_all", &wq_all, nl_dd_count) != 0)
    goto out;
  if (model_alloc_and_read(f, path, "wk_all", &wk_all, nl_dd_count) != 0)
    goto out;
  if (model_alloc_and_read(f, path, "wv_all", &wv_all, nl_dd_count) != 0)
    goto out;
  if (model_alloc_and_read(f, path, "wo_all", &wo_all, nl_dd_count) != 0)
    goto out;
  if (model_alloc_and_read(f, path, "rms_ffn_all", &rms_ffn_all, nl_d_count) !=
      0)
    goto out;
  if (model_alloc_and_read(f, path, "w1_all", &w1_all, nl_hdd_count) != 0)
    goto out;
  if (model_alloc_and_read(f, path, "w2_all", &w2_all, nl_dhd_count) != 0)
    goto out;
  if (model_alloc_and_read(f, path, "w3_all", &w3_all, nl_hdd_count) != 0)
    goto out;

  for (int l = 0; l < nl; l++) {
    size_t li = (size_t)l;
    rms_att_w[l] = (float *)malloc(d_count * sizeof(float));
    wq[l] = (float *)malloc(dd_count * sizeof(float));
    wk[l] = (float *)malloc(dd_count * sizeof(float));
    wv[l] = (float *)malloc(dd_count * sizeof(float));
    wo[l] = (float *)malloc(dd_count * sizeof(float));
    rms_ffn_w[l] = (float *)malloc(d_count * sizeof(float));
    w1[l] = (float *)malloc(hdd_count * sizeof(float));
    w2[l] = (float *)malloc(dhd_count * sizeof(float));
    w3[l] = (float *)malloc(hdd_count * sizeof(float));
    if (!rms_att_w[l] || !wq[l] || !wk[l] || !wv[l] || !wo[l] ||
        !rms_ffn_w[l] || !w1[l] || !w2[l] || !w3[l]) {
      fprintf(stderr, "Out of memory allocating layer %d tensors\n", l);
      goto out;
    }

    memcpy(rms_att_w[l], rms_att_all + li * d_count, d_count * sizeof(float));
    memcpy(wq[l], wq_all + li * dd_count, dd_count * sizeof(float));
    memcpy(wk[l], wk_all + li * dd_count, dd_count * sizeof(float));
    memcpy(wv[l], wv_all + li * dd_count, dd_count * sizeof(float));
    memcpy(wo[l], wo_all + li * dd_count, dd_count * sizeof(float));
    memcpy(rms_ffn_w[l], rms_ffn_all + li * d_count, d_count * sizeof(float));
    memcpy(w1[l], w1_all + li * hdd_count, hdd_count * sizeof(float));
    memcpy(w2[l], w2_all + li * dhd_count, dhd_count * sizeof(float));
    memcpy(w3[l], w3_all + li * hdd_count, hdd_count * sizeof(float));
  }

  free(rms_att_all);
  rms_att_all = NULL;
  free(wq_all);
  wq_all = NULL;
  free(wk_all);
  wk_all = NULL;
  free(wv_all);
  wv_all = NULL;
  free(wo_all);
  wo_all = NULL;
  free(rms_ffn_all);
  rms_ffn_all = NULL;
  free(w1_all);
  w1_all = NULL;
  free(w2_all);
  w2_all = NULL;
  free(w3_all);
  w3_all = NULL;

  if (model_alloc_and_read(f, path, "rms_final_w", &rms_final_w, d_count) != 0)
    goto out;
  if (shared) {
    wcls = token_embedding;
  } else if (model_alloc_and_read(f, path, "wcls", &wcls, token_count) != 0) {
    goto out;
  }

  if (model_close_checked(f, path) != 0) {
    f = NULL;
    goto out;
  }
  f = NULL;

  m->cfg = cfg;
  m->token_embedding = token_embedding;
  for (int l = 0; l < nl; l++) {
    m->rms_att_w[l] = rms_att_w[l];
    m->wq[l] = wq[l];
    m->wk[l] = wk[l];
    m->wv[l] = wv[l];
    m->wo[l] = wo[l];
    m->rms_ffn_w[l] = rms_ffn_w[l];
    m->w1[l] = w1[l];
    m->w2[l] = w2[l];
    m->w3[l] = w3[l];
  }
  m->rms_final_w = rms_final_w;
  m->wcls = wcls;
  rc = 0;

out:
  if (f && model_close_checked(f, path) != 0)
    rc = -1;
  if (rc != 0) {
    for (int l = 0; l < N_LAYERS; l++) {
      free(rms_att_w[l]);
      free(wq[l]);
      free(wk[l]);
      free(wv[l]);
      free(wo[l]);
      free(rms_ffn_w[l]);
      free(w1[l]);
      free(w2[l]);
      free(w3[l]);
    }
    free(token_embedding);
    if (wcls && wcls != token_embedding)
      free(wcls);
    free(rms_final_w);
  }
  free(rms_att_all);
  free(wq_all);
  free(wk_all);
  free(wv_all);
  free(wo_all);
  free(rms_ffn_all);
  free(w1_all);
  free(w2_all);
  free(w3_all);
  return rc;
}

// Compile a single baked-weight conv kernel
static ANEKernel *compile_conv_kernel(const float *weights, int in_ch,
                                      int out_ch, int spatial) {
  NSData *wb = mil_build_weight_blob(weights, out_ch, in_ch);
  NSString *mil = mil_gen_conv(in_ch, out_ch, spatial);
  size_t inBytes = (size_t)in_ch * spatial * 4;
  size_t outBytes = (size_t)out_ch * spatial * 4;
  return ane_compile([mil dataUsingEncoding:NSUTF8StringEncoding], wb, 1,
                     &inBytes, 1, &outBytes);
}

// Compile all per-layer ANE kernels with current weights
static int model_compile_kernels(Model *m, int seq_len) {
  m->seq_len = seq_len;
  int d = m->cfg.dim, hd = m->cfg.hidden_dim, vs = m->cfg.vocab_size;
  int S = seq_len;
  printf("Compiling %d ANE conv kernels (S=%d)...\n", N_LAYERS * 7 + 1, S);

  for (int l = 0; l < N_LAYERS; l++) {
    m->kern_q[l] = compile_conv_kernel(m->wq[l], d, d, S);
    m->kern_k[l] = compile_conv_kernel(m->wk[l], d, d, S);
    m->kern_v[l] = compile_conv_kernel(m->wv[l], d, d, S);
    m->kern_o[l] = compile_conv_kernel(m->wo[l], d, d, S);
    m->kern_w1[l] = compile_conv_kernel(m->w1[l], d, hd, S);
    m->kern_w2[l] = compile_conv_kernel(m->w2[l], hd, d, S);
    m->kern_w3[l] = compile_conv_kernel(m->w3[l], d, hd, S);
    if (!m->kern_q[l]) {
      fprintf(stderr, "L%d kern_q fail\n", l);
      return -1;
    }
    if (!m->kern_k[l]) {
      fprintf(stderr, "L%d kern_k fail\n", l);
      return -1;
    }
    if (!m->kern_v[l]) {
      fprintf(stderr, "L%d kern_v fail\n", l);
      return -1;
    }
    if (!m->kern_o[l]) {
      fprintf(stderr, "L%d kern_o fail\n", l);
      return -1;
    }
    if (!m->kern_w1[l]) {
      fprintf(stderr, "L%d kern_w1 fail\n", l);
      return -1;
    }
    if (!m->kern_w2[l]) {
      fprintf(stderr, "L%d kern_w2 fail\n", l);
      return -1;
    }
    if (!m->kern_w3[l]) {
      fprintf(stderr, "L%d kern_w3 fail\n", l);
      return -1;
    }
    printf("  Layer %d OK\n", l);
  }
  m->kern_cls = compile_conv_kernel(m->wcls, d, vs, S);
  if (!m->kern_cls) {
    fprintf(stderr,
            "Classifier kernel compile failed (dim=%d→vocab=%d too large?), "
            "using CPU for cls\n",
            d, vs);
  }
  printf("  All kernels compiled (%d conv + %s)\n", N_LAYERS * 7,
         m->kern_cls ? "cls" : "cls=CPU");
  return 0;
}

// Recompile all kernels after weight update — unload all first to avoid ANE
// model limit
static int model_recompile_kernels(Model *m) {
  int d = m->cfg.dim, hd = m->cfg.hidden_dim, vs = m->cfg.vocab_size;
  int S = m->seq_len;
  // Phase 1: unload+free all
  for (int l = 0; l < N_LAYERS; l++) {
    ane_free(m->kern_q[l]);
    ane_free(m->kern_k[l]);
    ane_free(m->kern_v[l]);
    ane_free(m->kern_o[l]);
    ane_free(m->kern_w1[l]);
    ane_free(m->kern_w2[l]);
    ane_free(m->kern_w3[l]);
    m->kern_q[l] = m->kern_k[l] = m->kern_v[l] = m->kern_o[l] = NULL;
    m->kern_w1[l] = m->kern_w2[l] = m->kern_w3[l] = NULL;
  }
  if (m->kern_cls) {
    ane_free(m->kern_cls);
    m->kern_cls = NULL;
  }
  // Phase 2: recompile all
  for (int l = 0; l < N_LAYERS; l++) {
    m->kern_q[l] = compile_conv_kernel(m->wq[l], d, d, S);
    m->kern_k[l] = compile_conv_kernel(m->wk[l], d, d, S);
    m->kern_v[l] = compile_conv_kernel(m->wv[l], d, d, S);
    m->kern_o[l] = compile_conv_kernel(m->wo[l], d, d, S);
    m->kern_w1[l] = compile_conv_kernel(m->w1[l], d, hd, S);
    m->kern_w2[l] = compile_conv_kernel(m->w2[l], hd, d, S);
    m->kern_w3[l] = compile_conv_kernel(m->w3[l], d, hd, S);
    if (!m->kern_q[l] || !m->kern_k[l] || !m->kern_v[l] || !m->kern_o[l] ||
        !m->kern_w1[l] || !m->kern_w2[l] || !m->kern_w3[l])
      return -1;
  }
  m->kern_cls = compile_conv_kernel(m->wcls, d, vs, S);
  // cls may fail for large vocab — that's OK, forward uses CPU fallback
  return 0;
}

static void model_alloc_training(Model *m) {
  int d = m->cfg.dim, hd = m->cfg.hidden_dim, vs = m->cfg.vocab_size,
      S = m->seq_len;
  for (int l = 0; l < N_LAYERS; l++) {
    m->act_x[l] = (float *)calloc(S * d, sizeof(float));
    m->act_xnorm[l] = (float *)calloc(S * d, sizeof(float));
    m->act_q[l] = (float *)calloc(S * d, sizeof(float));
    m->act_k[l] = (float *)calloc(S * d, sizeof(float));
    m->act_v[l] = (float *)calloc(S * d, sizeof(float));
    m->act_attn_out[l] = (float *)calloc(S * d, sizeof(float));
    m->act_ffn_in[l] = (float *)calloc(S * d, sizeof(float));
    m->act_h1[l] = (float *)calloc(S * hd, sizeof(float));
    m->act_h3[l] = (float *)calloc(S * hd, sizeof(float));
    m->act_silu[l] = (float *)calloc(S * hd, sizeof(float));

    m->grad_wq[l] = (float *)calloc(d * d, sizeof(float));
    m->grad_wk[l] = (float *)calloc(d * d, sizeof(float));
    m->grad_wv[l] = (float *)calloc(d * d, sizeof(float));
    m->grad_wo[l] = (float *)calloc(d * d, sizeof(float));
    m->grad_w1[l] = (float *)calloc(hd * d, sizeof(float));
    m->grad_w2[l] = (float *)calloc(d * hd, sizeof(float));
    m->grad_w3[l] = (float *)calloc(hd * d, sizeof(float));
  }
  m->act_final = (float *)calloc(S * d, sizeof(float));
  m->act_pre_final = (float *)calloc(S * d, sizeof(float));
  m->logits = (float *)calloc(S * vs, sizeof(float));
  m->grad_wcls = (float *)calloc(vs * d, sizeof(float));
  m->grad_emb = (float *)calloc(vs * d, sizeof(float));

  m->total_params = 0;
  for (int l = 0; l < N_LAYERS; l++)
    m->total_params += 4 * (size_t)d * d + 2 * (size_t)hd * d + (size_t)d * hd;
  m->total_params += (size_t)vs * d * 2;
  m->adam_m = (float *)calloc(m->total_params, sizeof(float));
  m->adam_v = (float *)calloc(m->total_params, sizeof(float));
  m->adam_step = 0;
  printf("Total trainable params: %zu (%.1f M)\n", m->total_params,
         m->total_params / 1e6);
}
