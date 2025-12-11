#ifndef BACKWARD_KERNELS_H
#define BACKWARD_KERNELS_H

#include <cuda_runtime.h>

// Conv2D Backward Kernels
__global__ void conv2d_backward_input(
    const float* grad_output,
    const float* weights,
    float* grad_input,
    int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int K
);

__global__ void conv2d_backward_weights(
    const float* grad_output,
    const float* input,
    float* grad_weights,
    int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int K
);

// ReLU Backward
__global__ void relu_backward(
    const float* grad_output,
    const float* input_before_relu,
    float* grad_input,
    int N
);

// Upsample Backward
__global__ void upsample_backward(
    const float* grad_output,
    float* grad_input,
    int H_in, int W_in, int C
);

#endif

