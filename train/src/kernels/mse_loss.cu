#include "../../include/mse_loss.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
using std::min;

// Simple kernel without shared memory to avoid PTX issues
__global__ void mse_loss_forward_kernel(
    const float* output,
    const float* target,
    float* partial_sum,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    // Each thread processes multiple elements
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride) {
        float d = output[i] - target[i];
        sum += d * d;
    }
    
    // Use atomicAdd to accumulate (no shared memory needed)
    if (sum != 0.0f) {
        atomicAdd(partial_sum, sum);
    }
}

float mse_loss_forward(
    const float* d_output,
    const float* d_target,
    int N
) {
    if (N == 0) return 0.0f;
    
    // Allocate single float for atomic accumulation
    float* d_sum;
    cudaError_t err = cudaMalloc(&d_sum, sizeof(float));
    if (err != cudaSuccess) {
        printf("[ERROR] cudaMalloc failed in mse_loss_forward: %s\n", cudaGetErrorString(err));
        return 0.0f;
    }
    
    // Initialize to zero
    cudaMemset(d_sum, 0, sizeof(float));
    
    // Launch kernel with simple configuration (no shared memory)
    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;
    
    mse_loss_forward_kernel<<<num_blocks, block_size>>>(
        d_output, d_target, d_sum, N
    );
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[ERROR] Kernel launch failed in mse_loss_forward: %s\n", cudaGetErrorString(err));
        cudaFree(d_sum);
        return 0.0f;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[ERROR] cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_sum);
        return 0.0f;
    }
    
    // Copy result back
    float total_sum = 0.0f;
    err = cudaMemcpy(&total_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("[ERROR] cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_sum);
        return 0.0f;
    }
    
    cudaFree(d_sum);
    
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

