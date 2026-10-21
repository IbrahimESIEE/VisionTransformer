#ifndef MODEL_PARTS_H
#define MODEL_PARTS_H

#include "types.h"   // definit data_t (float par defaut, ap_fixed si USE_FIXED_POINT=1)
#include "arch.h"    // definit ARCH (ITERATIF / PIPELINE) et BATCH_MAX



void layer_norm(const data_t* input, const data_t* gamma, const data_t* beta, data_t* output, int dim);
void relu(data_t* x, int size);
void matmul(const data_t* A, const data_t* B, data_t* C, int M, int K, int N); // C = A(MxK) @ B(KxN)
void add_vectors(const data_t* a, const data_t* b, data_t* out, int size);

void attention_core(const data_t* Q, const data_t* K, const data_t* V, data_t* out, int seq_len, int head_dim);
void feed_forward(const data_t* input, const data_t* W1, const data_t* b1, const data_t* W2, const data_t* b2, data_t* output, int embed_dim, int ff_dim);
void transformer_block(const data_t* input,
                       const data_t* Wq, const data_t* Wk, const data_t* Wv, const data_t* Wo, const data_t* bo,
                       const data_t* W1, const data_t* b1, const data_t* W2, const data_t* b2,
                       const data_t* norm1_g, const data_t* norm1_b, const data_t* norm2_g, const data_t* norm2_b,
                       data_t* output, int seq_len, int embed_dim, int num_heads);

void patch_embed(const data_t* image, const data_t* W, const data_t* b, data_t* out,
                 int Hp, int Wp, int patch_size, int in_channels, int embed_dim);
void pos_embed(const data_t* row_table, const data_t* col_table, data_t* out,
               int Hp, int Wp, int embed_dim);
void depth_head(const data_t* input, const data_t* W, const data_t* b, data_t* out,
                int Hp, int Wp, int patch_size, int embed_dim);

#endif
