#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "types.h"  
#include "arch.h"    

// void softmax_float(const float* input, float* output, int size);
void softmax_float(const data_t* input, data_t* output, int size);   // [AJOUT] data_t

#endif

