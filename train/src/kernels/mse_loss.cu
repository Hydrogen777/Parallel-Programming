#include "../../include/mse_loss.h"
#include <cuda_runtime.h>
#include <cstdio>

// Compute MSE loss on host to avoid PTX toolchain issues
float mse_loss_forward(
    const float* d_output,
    const float* d_target,
    int N
) {
    if (N == 0) return 0.0f;
    
    // Copy data from device to host
    float* h_output = new float[N];
    float* h_target = new float[N];
    
    cudaError_t err = cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("[ERROR] cudaMemcpy output failed: %s\n", cudaGetErrorString(err));
        delete[] h_output;
        delete[] h_target;
        return 0.0f;
    }
    
    err = cudaMemcpy(h_target, d_target, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("[ERROR] cudaMemcpy target failed: %s\n", cudaGetErrorString(err));
        delete[] h_output;
        delete[] h_target;
        return 0.0f;
    }
    
    // Compute MSE on host
    float total_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float d = h_output[i] - h_target[i];
        total_sum += d * d;
    }
    
    delete[] h_output;
    delete[] h_target;
    
    return total_sum / N;
}

__global__ void mse_loss_backward_kernel(
    const float* output,
    const float* target,
    float* grad_output,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        grad_output[idx] = 2.0f * (output[idx] - target[idx]) / N;
    }
}

void mse_loss_backward(
    const float* d_output,
    const float* d_target,
    float* d_grad_output,
    int N
) {
    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;
    
    mse_loss_backward_kernel<<<num_blocks, block_size>>>(
        d_output, d_target, d_grad_output, N
    );
    cudaDeviceSynchronize();
}

