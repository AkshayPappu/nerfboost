#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA Kernel for ReLU Activation
__global__ void relu_activation(float* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    x[idx] = fmaxf(0.0f, x[idx]);
}

// CUDA Kernel for Matrix Multiplication
__global__ void matmul_kernel(const float* A, const float* W, float* output, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < K) {
        float value = 0.0f;
        for (int i = 0; i < M; i++) {
            value += A[row * M + i] * W[i * K + col];
        }
        output[row * K + col] = value;
    }
}

// Helper function for matrix multiplication
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor W) {
    int N = A.size(0); // number of rows in A
    int M = A.size(1); // number of columns in A
    int K = W.size(1); // number of columns in W

    auto output = torch::zeros({N, K}, torch::device(A.device()).dtype(A.dtype()));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((K + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        W.data_ptr<float>(),
        output.data_ptr<float>(),
        N, M, K
    );

    return output;
}

// Helper function for ReLU activation
torch::Tensor relu_cuda(torch::Tensor x) {
    int size = x.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    relu_activation<<<blocks, threads>>>(x.data_ptr<float>(), size);

    return x;
}

// MLP Network using CUDA
torch::Tensor mlp_network_cuda(torch::Tensor sampled_points, torch::Tensor directions, std::vector<torch::Tensor> weights) {
    // Concatenate sampled points and directions (N, 6)
    auto inputs = torch::cat({sampled_points, directions}, -1);
    auto x = inputs;

    // Process through MLP layers
    for (const auto& W : weights) {
        x = matmul_cuda(x, W);  // Matrix multiplication
        x = relu_cuda(x);       // ReLU activation
    }

    cudaDeviceSynchronize();

    // Handle errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    
    return x;
}