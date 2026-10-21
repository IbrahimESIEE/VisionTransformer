#include <iostream>
#include <vector>
#include "../codes/model_parts.h"
#include "../codes/tensor_io.h"
#include "../codes/config.h"   // <-- DATA_DIR doit etre defini ici
#include "../params.h"          // [DDR-WEIGHTS] fournit le tableau global WEIGHTS[] (valeurs des poids)


#if ARCH == ARCH_PIPELINE// [MULTI-ARCH fix 2026-07-16] pipeline : 3 ports poids distincts (meme buffer DDR)
void vitis_top(const data_t* image_in,
               const data_t* weights_in0, const data_t* weights_in1, const data_t* weights_in2,
               data_t* depth_out, int num_images);
#else
void vitis_top(const data_t* image_in, const data_t* weights_in, data_t* depth_out, int num_images);
#endif



#ifndef DATA_DIR
#define DATA_DIR "D:/Devoir/Stage1/transformer/transformerCodeCppv2/hls_export/" //à modifier, bien choisir le chemin d'accès
#endif


static const std::string REF = DATA_DIR "reference/";

int main() {
    static float image[IN_CHANNELS][IMAGE_SIZE * IMAGE_SIZE];
    // static float depth_out[IMAGE_SIZE][IMAGE_SIZE];
    static data_t depth_out[IMAGE_SIZE][IMAGE_SIZE];   // [AJOUT] data_t (sortie du module)
    static float ref[IMAGE_SIZE][IMAGE_SIZE];
    static data_t image_q[IN_CHANNELS][IMAGE_SIZE * IMAGE_SIZE];   // [AJOUT] image convertie en data_t

    load_matrix<IN_CHANNELS, IMAGE_SIZE * IMAGE_SIZE>(REF + "input_image.txt",        image);
    load_matrix<IMAGE_SIZE, IMAGE_SIZE>            (REF + "final_depth_output.txt",   ref);

    // [AJOUT] conversion float -> data_t de l'image (quantification de l'entree)
    for (int c = 0; c < IN_CHANNELS; ++c)
        for (int k = 0; k < IMAGE_SIZE * IMAGE_SIZE; ++k)
            image_q[c][k] = (data_t)image[c][k];



#if ARCH == ARCH_PIPELINE
    // pipeline : les 3 ports poids pointent vers le MEME buffer simule &WEIGHTS[0]
    vitis_top(&image_q[0][0], &WEIGHTS[0], &WEIGHTS[0], &WEIGHTS[0], &depth_out[0][0], 1);
#else
    vitis_top(&image_q[0][0], &WEIGHTS[0], &depth_out[0][0], 1);   // [MULTI-ARCH] 1 image (validation csim)
#endif


    std::vector<float> out_flat, ref_flat;
    for (int r = 0; r < IMAGE_SIZE; r++)
        for (int c = 0; c < IMAGE_SIZE; c++) {
            // out_flat.push_back(depth_out[r][c]);
            out_flat.push_back((float)depth_out[r][c]);   // [AJOUT] cast data_t -> float pour comparer
            ref_flat.push_back(ref[r][c]);
        }

    
    
    bool ok = compare_tensors(out_flat, ref_flat, "TEST Modele Complet (vitis_top, local g++)", 1.5f);
    return ok ? 0 : 1;
}

