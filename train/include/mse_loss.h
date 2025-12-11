#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include <cuda_runtime.h>

float mse_loss_forward(
    const float* d_output,
    const float* d_target,
    int N
);

void mse_loss_backward(
    const float* d_output,
    const float* d_target,
    float* d_grad_output,
    int N
);

#endif

