#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA Kernel for volume rendering
__global__ void volume_rendering_kernel(const float* densities, const float* colors, const float* distances, int num_rays, int num_samples, float* final_color) 
{
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= num_rays) return;

    float transmittance = 1.0f;
    float color_r = 0.0f, color_g = 0.0f, color_b = 0.0f;

    for (int i = 0; i < num_samples; i++) {
        float density = densities[ray_idx * num_samples + i];
        float distance = distances[ray_idx * num_samples + i];

        float alpha = 1.0f - expf(-density * distance);

        float weight = alpha * transmittance;

        color_r += weight * colors[(ray_idx * num_samples + i) * 3 + 0];
        color_g += weight * colors[(ray_idx * num_samples + i) * 3 + 1];
        color_b += weight * colors[(ray_idx * num_samples + i) * 3 + 2];

        transmittance *= (1.0f - alpha + 1e-10f);
    }

    final_color[ray_idx * 3 + 0] = color_r;
    final_color[ray_idx * 3 + 1] = color_g;
    final_color[ray_idx * 3 + 2] = color_b;
}

// C++ Interface for CUDA volume rendering
torch::Tensor volume_rendering_cuda(torch::Tensor densities, torch::Tensor colors, torch::Tensor distances) {
    const int num_rays = densities.size(0);
    const int num_samples = densities.size(1);

    auto final_color = torch::zeros({num_rays, 3}, torch::device(densities.device()).dtype(densities.dtype()));

    const int threads_per_block = 256;
    const int num_blocks = (num_rays + threads_per_block - 1) / threads_per_block;

    volume_rendering_kernel<<<num_blocks, threads_per_block>>>(
        densities.data_ptr<float>(),
        colors.data_ptr<float>(),
        distances.data_ptr<float>(),
        num_rays,
        num_samples,
        final_color.data_ptr<float>()
    );

    cudaDeviceSynchronize();

    // handle errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    return final_color;
}
