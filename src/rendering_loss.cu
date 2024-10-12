#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for computing the squared differences
__global__ void mse_loss_kernel(const float* rendered_image, const float* ground_truth_image, float* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float diff = rendered_image[idx] - ground_truth_image[idx];
    atomicAdd(loss, diff * diff); 
}

// CUDA wrapper for computing MSE loss
torch::Tensor rendering_loss_cuda(torch::Tensor rendered_image, torch::Tensor ground_truth_image) {
    TORCH_CHECK(rendered_image.sizes() == ground_truth_image.sizes(), "Images must have the same dimensions");

    int size = rendered_image.numel();
    auto loss = torch::zeros(1, torch::device(rendered_image.device()).dtype(rendered_image.dtype()));

    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;

    mse_loss_kernel<<<num_blocks, threads_per_block>>>(
        rendered_image.data_ptr<float>(),
        ground_truth_image.data_ptr<float>(),
        loss.data_ptr<float>(),
        size
    );

    // Ensure the CUDA kernel execution is completed
    cudaDeviceSynchronize();

    // handle errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    // Compute the mean loss (divide by number of elements)
    return loss / size;
}