#include "model_parts.h"
#include "softmax.h"
#include "config.h"      // dimensions fixes : NUM_PATCHES, EMBED_DIM, HEAD_DIM, FF_DIM...
#include <cmath>
#include <algorithm>



#define MAX_SEQ         NUM_PATCHES
#define MAX_EMBED       EMBED_DIM
#define MAX_HEAD        HEAD_DIM
#define MAX_FF          FF_DIM
#define MAX_PATCH_PIX   (PATCH_SIZE * PATCH_SIZE * IN_CHANNELS)   // patch_embed
#define MAX_DEPTH_PIX   (PATCH_SIZE * PATCH_SIZE)                 // depth_head


void layer_norm(const data_t* input, const data_t* gamma, const data_t* beta, data_t* output, int dim) {   
    acc_t mean = 0;         // [FIXED-ACC]
    for (int i = 0; i < dim; ++i) mean += (acc_t)input[i];   // [FIXED-ACC] accumulation en virgule fixe
    mean /= dim;
    acc_t var = 0;         // [FIXED-ACC]
    for (int i = 0; i < dim; ++i) { acc_t d = (acc_t)input[i] - mean; var += d * d; }   // [FIXED-ACC] ecart en virgule fixe
    var /= dim;

    float eps = 1e-5f;
    float std_inv = 1.0f / sqrtf((float)var + eps);   // [FIXED-ACC]

    for (int i = 0; i < dim; ++i) {
        output[i] = (data_t)( (acc_t)gamma[i] * ((acc_t)input[i] - mean) * (acc_t)std_inv + (acc_t)beta[i] );   // [FIXED-ACC] combinaison lineaire en virgule fixe
    }
}

void relu(data_t* x, int size) {   // [AJOUT] data_t
    for (int i = 0; i < size; ++i) x[i] = (x[i] > (data_t)0) ? x[i] : (data_t)0;   // [FIXED-ACC] comparaison en virgule fixe (plus de fcmp flottant)
}

void add_vectors(const data_t* a, const data_t* b, data_t* out, int size) {   // [AJOUT] data_t
    for (int i = 0; i < size; ++i) out[i] = (data_t)((acc_t)a[i] + (acc_t)b[i]);   // [FIXED-ACC] addition en virgule fixe
}

void matmul(const data_t* A, const data_t* B, data_t* C, int M, int K, int N) {   // [AJOUT] data_t
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            acc_t sum = 0;         // [FIXED-ACC]
            for (int k = 0; k < K; ++k) {
                sum += (acc_t)(A[m * K + k] * B[k * N + n]);   // [FIXED-ACC] produit data_t*data_t + accumulation en virgule fixe
            }
            C[m * N + n] = (data_t)sum;   // [AJOUT] stockage data_t
        }
    }
}

void attention_core(const data_t* Q, const data_t* K, const data_t* V, data_t* out, int seq_len, int head_dim) {   // [AJOUT] data_t
    float scale = 1.0f / sqrtf((float)head_dim);
    data_t scores[MAX_SEQ * MAX_SEQ];        // [MULTI-ARCH] non-static : memoire propre par instance (etage pipeline)
    data_t soft_scores[MAX_SEQ * MAX_SEQ];   // [MULTI-ARCH] non-static

    matmul(Q, K, scores, seq_len, head_dim, seq_len);

    for (int i = 0; i < seq_len * seq_len; ++i) scores[i] = (data_t)((acc_t)scores[i] * (acc_t)scale);   // [FIXED-ACC] mise a l'echelle en virgule fixe

    for (int i = 0; i < seq_len; ++i) {
        softmax_float(&scores[i * seq_len], &soft_scores[i * seq_len], seq_len);
    }

    matmul(soft_scores, V, out, seq_len, seq_len, head_dim);
}

void feed_forward(const data_t* input, const data_t* W1, const data_t* b1, const data_t* W2, const data_t* b2, data_t* output, int embed_dim, int ff_dim) {   // [AJOUT] data_t
    data_t hidden[MAX_FF];   // [MULTI-ARCH] non-static : memoire propre par instance (etage pipeline)
    matmul(input, W1, hidden, 1, embed_dim, ff_dim);
    for (int i = 0; i < ff_dim; ++i) hidden[i] = (data_t)((acc_t)hidden[i] + (acc_t)b1[i]);   // [FIXED-ACC] biais en virgule fixe
    relu(hidden, ff_dim);

    matmul(hidden, W2, output, 1, ff_dim, embed_dim);
    for (int i = 0; i < embed_dim; ++i) output[i] = (data_t)((acc_t)output[i] + (acc_t)b2[i]);   // [FIXED-ACC] biais en virgule fixe
}
void transformer_block(const data_t* input,
                       const data_t* Wq, const data_t* Wk, const data_t* Wv, const data_t* Wo, const data_t* bo,
                       const data_t* W1, const data_t* b1, const data_t* W2, const data_t* b2,
                       const data_t* norm1_g, const data_t* norm1_b, const data_t* norm2_g, const data_t* norm2_b,
                       data_t* output, int seq_len, int embed_dim, int num_heads) {   // [AJOUT] data_t

#if (ARCH == ARCH_PIPELINE) || (ARCH == ARCH_SERIE)
    #pragma HLS INLINE recursive   // pipeline ET serie : chaque bloc = materiel distinct
#elif ARCH == ARCH_ITERATIF
    #pragma HLS INLINE off          // iteratif : 1 seule instance reutilisee (folding)
#endif

    int head_dim = embed_dim / num_heads;
    data_t Q[MAX_SEQ * MAX_EMBED], K[MAX_SEQ * MAX_EMBED], V[MAX_SEQ * MAX_EMBED];   // [MULTI-ARCH] non-static
    data_t attn_out[MAX_SEQ * MAX_EMBED];      // [MULTI-ARCH] non-static
    data_t concat_heads[MAX_SEQ * MAX_EMBED];  // [MULTI-ARCH] non-static

    matmul(input, Wq, Q, seq_len, embed_dim, embed_dim);
    matmul(input, Wk, K, seq_len, embed_dim, embed_dim);
    matmul(input, Wv, V, seq_len, embed_dim, embed_dim);

    for (int h = 0; h < num_heads; ++h) {
        data_t Qh[MAX_SEQ * MAX_HEAD], Kh[MAX_SEQ * MAX_HEAD], Vh[MAX_SEQ * MAX_HEAD];   // [MULTI-ARCH] non-static
        data_t Kh_T[MAX_HEAD * MAX_SEQ];   // [MULTI-ARCH] non-static
        data_t out_h[MAX_SEQ * MAX_HEAD];  // [MULTI-ARCH] non-static

        for (int s = 0; s < seq_len; ++s) {
            for (int d = 0; d < head_dim; ++d) {
                Qh[s * head_dim + d] = Q[s * embed_dim + h * head_dim + d];
                Kh[s * head_dim + d] = K[s * embed_dim + h * head_dim + d];
                Vh[s * head_dim + d] = V[s * embed_dim + h * head_dim + d];
            }
        }
        for (int s = 0; s < seq_len; ++s)
            for (int d = 0; d < head_dim; ++d)
                Kh_T[d * seq_len + s] = Kh[s * head_dim + d];

        attention_core(Qh, Kh_T, Vh, out_h, seq_len, head_dim);

        for (int s = 0; s < seq_len; ++s)
            for (int d = 0; d < head_dim; ++d)
                concat_heads[s * embed_dim + h * head_dim + d] = out_h[s * head_dim + d];
    }

    matmul(concat_heads, Wo, attn_out, seq_len, embed_dim, embed_dim);
    for (int s = 0; s < seq_len; ++s)
        for (int e = 0; e < embed_dim; ++e)
            attn_out[s * embed_dim + e] = (data_t)((acc_t)attn_out[s * embed_dim + e] + (acc_t)bo[e]);   // [FIXED-ACC] biais en virgule fixe
    data_t norm1_in[MAX_SEQ * MAX_EMBED];    // [MULTI-ARCH] non-static
    data_t norm1_out[MAX_SEQ * MAX_EMBED];   // [MULTI-ARCH] non-static
    add_vectors(input, attn_out, norm1_in, seq_len * embed_dim);
    for (int s = 0; s < seq_len; ++s) {
        layer_norm(&norm1_in[s * embed_dim], norm1_g, norm1_b, &norm1_out[s * embed_dim], embed_dim);
    }

    data_t ff_out[MAX_SEQ * MAX_EMBED];   // [MULTI-ARCH] non-static
    for (int s = 0; s < seq_len; ++s) {
        feed_forward(&norm1_out[s * embed_dim], W1, b1, W2, b2, &ff_out[s * embed_dim], embed_dim, embed_dim * 4);
    }

    data_t norm2_in[MAX_SEQ * MAX_EMBED];   // [MULTI-ARCH] non-static
    add_vectors(norm1_out, ff_out, norm2_in, seq_len * embed_dim);
    for (int s = 0; s < seq_len; ++s) {
        layer_norm(&norm2_in[s * embed_dim], norm2_g, norm2_b, &output[s * embed_dim], embed_dim);
    }
}

void patch_embed(const data_t* image, const data_t* W, const data_t* b, data_t* out,
                 int Hp, int Wp, int patch_size, int in_channels, int embed_dim) {   // [AJOUT] data_t
    int patch_pixels = patch_size * patch_size * in_channels;
    int H_img = Hp * patch_size;
    int W_img = Wp * patch_size;

    for (int i = 0; i < Hp; ++i) {
        for (int j = 0; j < Wp; ++j) {
            int patch_idx = i * Wp + j;
            data_t patch_vec[MAX_PATCH_PIX];   // [MULTI-ARCH] non-static (patch_embed appele par 1 seul etage : sans effet en iteratif)

            for (int c = 0; c < in_channels; ++c) {
                for (int py = 0; py < patch_size; ++py) {
                    for (int px = 0; px < patch_size; ++px) {
                        int img_row = i * patch_size + py;
                        int img_col = j * patch_size + px;
                        int p_idx = c * patch_size * patch_size + py * patch_size + px;
                        patch_vec[p_idx] = image[c * H_img * W_img + img_row * W_img + img_col];
                    }
                }
            }

            for (int e = 0; e < embed_dim; ++e) {
                acc_t sum = (acc_t)b[e];   // [FIXED-ACC] init biais en virgule fixe
                for (int p = 0; p < patch_pixels; ++p) {
                    sum += (acc_t)(patch_vec[p] * W[e * patch_pixels + p]);   // [FIXED-ACC] produit + accumulation en virgule fixe
                }
                out[patch_idx * embed_dim + e] = (data_t)sum;   // [AJOUT] stockage data_t
            }
        }
    }
}


void pos_embed(const data_t* row_table, const data_t* col_table, data_t* out,
               int Hp, int Wp, int embed_dim) {   // [AJOUT] data_t
    int half_dim = embed_dim / 2;
    for (int i = 0; i < Hp; ++i) {
        for (int j = 0; j < Wp; ++j) {
            int idx = i * Wp + j;
            for (int d = 0; d < half_dim; ++d) {
                out[idx * embed_dim + d] = row_table[i * half_dim + d];
                out[idx * embed_dim + half_dim + d] = col_table[j * half_dim + d];
            }
        }
    }
}


void depth_head(const data_t* input, const data_t* W, const data_t* b, data_t* out,
                int Hp, int Wp, int patch_size, int embed_dim) {   // [AJOUT] data_t
    int W_img = Wp * patch_size;
    int patch_pixels = patch_size * patch_size;

    for (int i = 0; i < Hp; ++i) {
        for (int j = 0; j < Wp; ++j) {
            int patch_idx = i * Wp + j;

            data_t patch_out[MAX_DEPTH_PIX];   // [MULTI-ARCH] non-static (depth_head appele par 1 seul etage : sans effet en iteratif)

            for (int p = 0; p < patch_pixels; ++p) {

                acc_t sum = (acc_t)b[p];   // [FIXED-ACC] init biais en virgule fixe
                for (int e = 0; e < embed_dim; ++e) {

                    sum += (acc_t)(input[patch_idx * embed_dim + e] * W[p * embed_dim + e]);   // [FIXED-ACC] produit + accumulation en virgule fixe
                }
                // patch_out[p] = sum;
                patch_out[p] = (data_t)sum;   // [AJOUT] stockage data_t
            }

            for (int py = 0; py < patch_size; ++py) {
                for (int px = 0; px < patch_size; ++px) {
                    int img_row = i * patch_size + py;
                    int img_col = j * patch_size + px;
                    out[img_row * W_img + img_col] = patch_out[py * patch_size + px];
                }
            }
        }
    }
}
