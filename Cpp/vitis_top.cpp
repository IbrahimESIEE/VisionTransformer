#include "model_parts.h"
#include "../params.h"
#include "config.h"

#define IMG_ELEMS   (IN_CHANNELS * IMAGE_SIZE * IMAGE_SIZE)   // 3*64*64 = 12288
#define DEPTH_ELEMS (IMAGE_SIZE * IMAGE_SIZE)                 // 64*64   = 4096
#define TOKENS      (NUM_PATCHES * EMBED_DIM)                 // 16*32   = 512  (1 image encodee)

#if (ARCH == ARCH_PIPELINE) && (NUM_BLOCKS != 3)
#error "Le pipeline a 3 etages suppose NUM_BLOCKS==3. Adapte pipe_stageX si NUM_BLOCKS change."
#endif


#if ARCH == ARCH_PIPELINE


#include <hls_stream.h>

// --- Etage 1 : image brute -> patch embed + pos embed + addition -> BLOC 0 -> flux
static void pipe_stage1(const data_t* image_in, const data_t* WEIGHTS,
                        hls::stream<data_t>& out, int num_images) {
    for (int im = 0; im < num_images; ++im) {
        data_t x[TOKENS];
        data_t pos[TOKENS];
        data_t bin[TOKENS];
        data_t bout[TOKENS];

        patch_embed(&image_in[im * IMG_ELEMS],
                    &WEIGHTS[OFFSET_PATCH_EMBED_W], &WEIGHTS[OFFSET_PATCH_EMBED_B],
                    x, HP, WP, PATCH_SIZE, IN_CHANNELS, EMBED_DIM);
        pos_embed(&WEIGHTS[OFFSET_POS_ROW_TABLE], &WEIGHTS[OFFSET_POS_COL_TABLE],
                  pos, HP, WP, EMBED_DIM);
        add_vectors(x, pos, bin, TOKENS);

        transformer_block(bin,
            &WEIGHTS[OFF_WQ[0]], &WEIGHTS[OFF_WK[0]], &WEIGHTS[OFF_WV[0]],
            &WEIGHTS[OFF_WO[0]], &WEIGHTS[OFF_BO[0]],
            &WEIGHTS[OFF_W1[0]], &WEIGHTS[OFF_B1[0]],
            &WEIGHTS[OFF_W2[0]], &WEIGHTS[OFF_B2[0]],
            &WEIGHTS[OFF_NORM1_G[0]], &WEIGHTS[OFF_NORM1_B[0]],
            &WEIGHTS[OFF_NORM2_G[0]], &WEIGHTS[OFF_NORM2_B[0]],
            bout, NUM_PATCHES, EMBED_DIM, NUM_HEADS);

        for (int i = 0; i < TOKENS; ++i) out.write(bout[i]);
    }
}

// --- Etage 2 : flux -> BLOC 1 -> flux
static void pipe_stage2(hls::stream<data_t>& in, const data_t* WEIGHTS,
                        hls::stream<data_t>& out, int num_images) {
    for (int im = 0; im < num_images; ++im) {
        data_t bin[TOKENS];
        data_t bout[TOKENS];

        for (int i = 0; i < TOKENS; ++i) bin[i] = in.read();

        transformer_block(bin,
            &WEIGHTS[OFF_WQ[1]], &WEIGHTS[OFF_WK[1]], &WEIGHTS[OFF_WV[1]],
            &WEIGHTS[OFF_WO[1]], &WEIGHTS[OFF_BO[1]],
            &WEIGHTS[OFF_W1[1]], &WEIGHTS[OFF_B1[1]],
            &WEIGHTS[OFF_W2[1]], &WEIGHTS[OFF_B2[1]],
            &WEIGHTS[OFF_NORM1_G[1]], &WEIGHTS[OFF_NORM1_B[1]],
            &WEIGHTS[OFF_NORM2_G[1]], &WEIGHTS[OFF_NORM2_B[1]],
            bout, NUM_PATCHES, EMBED_DIM, NUM_HEADS);

        for (int i = 0; i < TOKENS; ++i) out.write(bout[i]);
    }
}

// --- Etage 3 : flux -> BLOC 2 -> norm finale -> depth head -> DDR
static void pipe_stage3(hls::stream<data_t>& in, const data_t* WEIGHTS,
                        data_t* depth_out, int num_images) {
    for (int im = 0; im < num_images; ++im) {
        data_t bin[TOKENS];
        data_t bout[TOKENS];
        data_t norm_out[TOKENS];

        for (int i = 0; i < TOKENS; ++i) bin[i] = in.read();

        transformer_block(bin,
            &WEIGHTS[OFF_WQ[2]], &WEIGHTS[OFF_WK[2]], &WEIGHTS[OFF_WV[2]],
            &WEIGHTS[OFF_WO[2]], &WEIGHTS[OFF_BO[2]],
            &WEIGHTS[OFF_W1[2]], &WEIGHTS[OFF_B1[2]],
            &WEIGHTS[OFF_W2[2]], &WEIGHTS[OFF_B2[2]],
            &WEIGHTS[OFF_NORM1_G[2]], &WEIGHTS[OFF_NORM1_B[2]],
            &WEIGHTS[OFF_NORM2_G[2]], &WEIGHTS[OFF_NORM2_B[2]],
            bout, NUM_PATCHES, EMBED_DIM, NUM_HEADS);

        for (int s = 0; s < NUM_PATCHES; ++s) {
            layer_norm(&bout[s * EMBED_DIM],
                       &WEIGHTS[OFFSET_FINAL_NORM_G], &WEIGHTS[OFFSET_FINAL_NORM_B],
                       &norm_out[s * EMBED_DIM], EMBED_DIM);
        }
        depth_head(norm_out, &WEIGHTS[OFFSET_DEPTH_HEAD_W], &WEIGHTS[OFFSET_DEPTH_HEAD_B],
                   &depth_out[im * DEPTH_ELEMS], HP, WP, PATCH_SIZE, EMBED_DIM);
    }
}
#endif  // ARCH == ARCH_PIPELINE


#if ARCH == ARCH_PIPELINE
// ============================================================
//  PIPELINE : signature a 3 PORTS POIDS (fix DATAFLOW 2026-07-16)
//  Le DATAFLOW interdit qu'UN port m_axi soit lu par plusieurs etages
//  concurrents (erreur HLS 200-1013 : gmem2 read in multiple processes).
//  Solution : chaque etage a SON port poids (gmem2 / gmem3 / gmem4). Les 3
//  pointent vers le MEME buffer DDR -> cote PYNQ, on ecrit la MEME adresse
//  dans weights_in0, weights_in1 et weights_in2.
// ============================================================
void vitis_top(const data_t* image_in,
               const data_t* weights_in0, const data_t* weights_in1, const data_t* weights_in2,
               data_t* depth_out, int num_images) {
    #pragma HLS INTERFACE m_axi     port=image_in    offset=slave bundle=gmem0 depth=IMG_ELEMS
    #pragma HLS INTERFACE m_axi     port=weights_in0 offset=slave bundle=gmem2 depth=TOTAL_WEIGHTS
    #pragma HLS INTERFACE m_axi     port=weights_in1 offset=slave bundle=gmem3 depth=TOTAL_WEIGHTS
    #pragma HLS INTERFACE m_axi     port=weights_in2 offset=slave bundle=gmem4 depth=TOTAL_WEIGHTS
    #pragma HLS INTERFACE m_axi     port=depth_out   offset=slave bundle=gmem1 depth=DEPTH_ELEMS
    #pragma HLS INTERFACE s_axilite port=image_in    bundle=control
    #pragma HLS INTERFACE s_axilite port=weights_in0 bundle=control
    #pragma HLS INTERFACE s_axilite port=weights_in1 bundle=control
    #pragma HLS INTERFACE s_axilite port=weights_in2 bundle=control
    #pragma HLS INTERFACE s_axilite port=depth_out   bundle=control
    #pragma HLS INTERFACE s_axilite port=num_images  bundle=control
    #pragma HLS INTERFACE s_axilite port=return      bundle=control

    // -------- 3 etages concurrents relies par des flux --------
    #pragma HLS DATAFLOW
    hls::stream<data_t> s01;
    hls::stream<data_t> s12;
    #pragma HLS STREAM variable=s01 depth=2*TOKENS
    #pragma HLS STREAM variable=s12 depth=2*TOKENS

    // Chaque etage lit SON port poids (3 ports distincts -> meme buffer DDR)
    pipe_stage1(image_in, weights_in0, s01, num_images);       // patch/pos/add + bloc 0
    pipe_stage2(s01,      weights_in1, s12, num_images);       // bloc 1
    pipe_stage3(s12,      weights_in2, depth_out, num_images); // bloc 2 + norm + depth
}

#else
// ============================================================
//  ITERATIF ou SERIE : MEME fonction (1 seul port poids, pas de DATAFLOW).
//    - ITERATIF : boucle des blocs NON deroulee -> 1 instance reutilisee (folding).
//    - SERIE    : boucle des blocs DEROULEE (UNROLL) -> NUM_BLOCKS instances
//                 spatiales distinctes, sequentielles.
//  Interface identique => MEME block design / TCL, rien a changer dans Vivado.
// ============================================================
void vitis_top(const data_t* image_in, const data_t* weights_in, data_t* depth_out, int num_images) {
    #pragma HLS INTERFACE m_axi     port=image_in   offset=slave bundle=gmem0 depth=IMG_ELEMS
    #pragma HLS INTERFACE m_axi     port=weights_in offset=slave bundle=gmem2 depth=TOTAL_WEIGHTS
    #pragma HLS INTERFACE m_axi     port=depth_out  offset=slave bundle=gmem1 depth=DEPTH_ELEMS
    #pragma HLS INTERFACE s_axilite port=image_in   bundle=control
    #pragma HLS INTERFACE s_axilite port=weights_in bundle=control
    #pragma HLS INTERFACE s_axilite port=depth_out  bundle=control
    #pragma HLS INTERFACE s_axilite port=num_images bundle=control
    #pragma HLS INTERFACE s_axilite port=return     bundle=control

    // [DDR-WEIGHTS] Alias local : le nom "WEIGHTS" reste utilise partout.
    const data_t* WEIGHTS = weights_in;

    // -------- 1 transformer reutilise NUM_BLOCKS fois par image --------
    for (int im = 0; im < num_images; ++im) {
        data_t x[TOKENS];
        data_t pos[TOKENS];
        data_t block_in[TOKENS];
        data_t block_out[TOKENS];
        data_t block_out2[TOKENS];
        data_t norm_out[TOKENS];

        // 1. Patch embedding
        patch_embed(&image_in[im * IMG_ELEMS],
                    &WEIGHTS[OFFSET_PATCH_EMBED_W], &WEIGHTS[OFFSET_PATCH_EMBED_B],
                    x, HP, WP, PATCH_SIZE, IN_CHANNELS, EMBED_DIM);
        // 2. Positional embedding + addition
        pos_embed(&WEIGHTS[OFFSET_POS_ROW_TABLE], &WEIGHTS[OFFSET_POS_COL_TABLE],
                  pos, HP, WP, EMBED_DIM);
        add_vectors(x, pos, block_in, TOKENS);

        // 3. Blocs transformer : 1 seule instance materielle, reutilisee (folding).
        //    Ping-pong entre block_out / block_out2 ; les poids du bloc b viennent
        //    des tables OFF_*[b] (generees par generate_params_h.py).
        const data_t* cur = block_in;
        for (int b = 0; b < NUM_BLOCKS; ++b) {
        #if ARCH == ARCH_SERIE
            #pragma HLS UNROLL   // serie : deroule -> NUM_BLOCKS blocs spatiaux distincts
        #endif
            data_t* dst = (b & 1) ? block_out2 : block_out;
            transformer_block(cur,
                &WEIGHTS[OFF_WQ[b]], &WEIGHTS[OFF_WK[b]], &WEIGHTS[OFF_WV[b]],
                &WEIGHTS[OFF_WO[b]], &WEIGHTS[OFF_BO[b]],
                &WEIGHTS[OFF_W1[b]], &WEIGHTS[OFF_B1[b]],
                &WEIGHTS[OFF_W2[b]], &WEIGHTS[OFF_B2[b]],
                &WEIGHTS[OFF_NORM1_G[b]], &WEIGHTS[OFF_NORM1_B[b]],
                &WEIGHTS[OFF_NORM2_G[b]], &WEIGHTS[OFF_NORM2_B[b]],
                dst, NUM_PATCHES, EMBED_DIM, NUM_HEADS);
            cur = dst;   // la sortie du bloc b devient l'entree du bloc b+1
        }
        // 'cur' pointe sur la sortie du dernier bloc.

        // 4. Norm finale
        for (int s = 0; s < NUM_PATCHES; ++s) {
            layer_norm(&cur[s * EMBED_DIM],
                       &WEIGHTS[OFFSET_FINAL_NORM_G], &WEIGHTS[OFFSET_FINAL_NORM_B],
                       &norm_out[s * EMBED_DIM], EMBED_DIM);
        }
        // 5. Depth head
        depth_head(norm_out, &WEIGHTS[OFFSET_DEPTH_HEAD_W], &WEIGHTS[OFFSET_DEPTH_HEAD_B],
                   &depth_out[im * DEPTH_ELEMS], HP, WP, PATCH_SIZE, EMBED_DIM);
    }
}
#endif
