#ifndef SGD_OPTIMIZER_H
#define SGD_OPTIMIZER_H

#include <cuda_runtime.h>

void sgd_update(
    float* d_weights,
    const float* d_grad_weights,
    float learning_rate,
    int N
);

void zero_gradient(
    float* d_grad,
    int N
);

#endif

