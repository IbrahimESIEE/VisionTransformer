


#pragma once

template<int N, int IN_DIM, int OUT_DIM>
void dense(const float in[N][IN_DIM],
           const float weight[OUT_DIM][IN_DIM],
           const float bias[OUT_DIM],
           float out[N][OUT_DIM],
           bool has_bias) {
    for (int n = 0; n < N; n++) {
        for (int o = 0; o < OUT_DIM; o++) {
            float acc = has_bias ? bias[o] : 0.0f;
            for (int i = 0; i < IN_DIM; i++) {
                acc += in[n][i] * weight[o][i];
            }
            out[n][o] = acc;
        }
    }
}
