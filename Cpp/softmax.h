#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "types.h"   // [AJOUT] definit data_t
#include "arch.h"    // [MULTI-ARCH 2026-07-15] definit ARCH (pour l'inline en pipeline)

// void softmax_float(const float* input, float* output, int size);
void softmax_float(const data_t* input, data_t* output, int size);   // [AJOUT] data_t

#endif

// ------------------------------------------------------------------
// Modifie par Claude (Opus 4.8) le 2026-07-08 : prototype float -> data_t.
// ------------------------------------------------------------------
