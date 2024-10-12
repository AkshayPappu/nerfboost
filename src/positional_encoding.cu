#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA Kernel for Positional Encoding
__global__ void positional_encoding_kernel(float* x, float* encoded, int L, int x_size, int x_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= x_size * x_dim) return;

    int element_idx = idx / x_dim;  // Row index
    int dim_idx = idx % x_dim;      // Dimension index

    // Obtain the current value of x
    float value = x[element_idx * x_dim + dim_idx];  

    // Sine and cosine computations for each frequency
    for (int i = 0; i < L; i++) {
        float freq = powf(2.0f, i);  // Compute 2^i

        // Store sine and cosine values
        encoded[element_idx * (x_dim * 2 * L) + i * 2 * x_dim + dim_idx] = sinf(freq * value);       
        encoded[element_idx * (x_dim * 2 * L) + (i * 2 + 1) * x_dim + dim_idx] = cosf(freq * value);  
    }
}

// Launcher for the CUDA kernel
torch::Tensor positional_encoding_cuda(torch::Tensor x, int L) {
    int x_size = x.size(0);  
    int x_dim = x.size(1);   

    auto encoded = torch::zeros({x_size, x_dim * 2 * L}, torch::device(x.device()).dtype(x.dtype()));

    int threads_per_block = 256;
    int num_blocks = (x_size * x_dim + threads_per_block - 1) / threads_per_block;

    positional_encoding_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(),  
        encoded.data_ptr<float>(),  
        L,  
        x_size,  
        x_dim  
    );

    return encoded;
}