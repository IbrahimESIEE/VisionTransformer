#include "softmax.h"
#include <math.h>


void softmax_float(const data_t* input, data_t* output, int size) {   // [AJOUT] data_t
#if ARCH == ARCH_PIPELINE
    #pragma HLS INLINE   // [MULTI-ARCH] fondu dans chaque etage (DATAFLOW interdit le partage)
#endif
    if (size <= 0) return;

    // 1. Trouver le max pour la stabilite numerique
    // float max_val = input[0];
    float max_val = (float)input[0];   // [AJOUT] cast
    for (int i = 1; i < size; ++i) {
        // if (input[i] > max_val) {
        if ((float)input[i] > max_val) {   // [AJOUT] cast
            // max_val = input[i];
            max_val = (float)input[i];   // [AJOUT] cast
        }
    }

    // 2. Calculer les exponentielles et leur somme
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        // output[i] = expf(input[i] - max_val);
        output[i] = (data_t)expf((float)input[i] - max_val);   // [AJOUT] exp en float, stockage data_t
        // sum += output[i];
        sum += (float)output[i];   // [AJOUT] cast
    }

    // 3. Normaliser
    for (int i = 0; i < size; ++i) {
        // output[i] /= sum;
        output[i] = (data_t)((float)output[i] / sum);   // [AJOUT] cast
    }
}
