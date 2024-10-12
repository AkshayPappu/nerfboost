#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA Kernel for generating rays
__global__ void generate_rays_kernel(float* ray_origins, float* ray_directions, const float fx, const float fy, const float cx, const float cy, const int H, const int W) {
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= H * W) return;

    int i = pixel_idx % W;
    int j = pixel_idx / W;  

    float x = (i - cx) / fx;
    float y = (j - cy) / fy;
    float z = 1.0f;

    // Normalize the ray direction
    float norm = sqrtf(x * x + y * y + z * z);
    float dir_x = x / norm;
    float dir_y = y / norm;
    float dir_z = z / norm;

    ray_origins[pixel_idx * 3 + 0] = 0.0f;
    ray_origins[pixel_idx * 3 + 1] = 0.0f;
    ray_origins[pixel_idx * 3 + 2] = 0.0f;

    ray_directions[pixel_idx * 3 + 0] = dir_x;
    ray_directions[pixel_idx * 3 + 1] = dir_y;
    ray_directions[pixel_idx * 3 + 2] = dir_z;
}

// Wrapper function for the CUDA kernel
std::tuple<torch::Tensor, torch::Tensor> generate_rays_cuda(torch::Tensor camera_intrinsics, int H, int W) {
    float fx = camera_intrinsics[0][0].item<float>();
    float fy = camera_intrinsics[1][1].item<float>();
    float cx = camera_intrinsics[0][2].item<float>();
    float cy = camera_intrinsics[1][2].item<float>();

    // Create output tensors for ray origins and directions
    auto ray_origins = torch::zeros({H * W, 3}, torch::device(camera_intrinsics.device()).dtype(torch::kFloat32));
    auto ray_directions = torch::zeros({H * W, 3}, torch::device(camera_intrinsics.device()).dtype(torch::kFloat32));

    // Define thread layout for CUDA
    const int threads_per_block = 256;
    const int num_blocks = (H * W + threads_per_block - 1) / threads_per_block;

    // Launch the CUDA kernel
    generate_rays_kernel<<<num_blocks, threads_per_block>>>(
        ray_origins.data_ptr<float>(), ray_directions.data_ptr<float>(),
        fx, fy, cx, cy, H, W
    );

    // Ensure synchronization
    cudaDeviceSynchronize();

    return std::make_tuple(ray_origins, ray_directions);
}