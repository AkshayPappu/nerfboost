#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// CUDA Kernel for Deterministic Stratified Sampling
__global__ void stratified_sampling_kernel(float* t_vals, float* stratified_samples, int num_samples) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_samples - 1) {
        stratified_samples[i] = 0.5f * (t_vals[i] + t_vals[i + 1]);
    }
}

// Launcher for the CUDA kernel
torch::Tensor stratified_sampling_cuda(float near, float far, int num_samples) {
    auto stratified_samples = torch::zeros({num_samples - 1}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    auto t_vals = torch::linspace(near, far, num_samples, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    int threads_per_block = 256;
    int num_blocks = (num_samples - 1 + threads_per_block - 1) / threads_per_block;

    stratified_sampling_kernel<<<num_blocks, threads_per_block>>>(
        t_vals.data_ptr<float>(), stratified_samples.data_ptr<float>(), num_samples);

    cudaDeviceSynchronize();

    return stratified_samples;
}

// CUDA Kernel for Uniform Sampling
__global__ void uniform_sampling_kernel(float *t_vals, float near, float far, int num_samples) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_samples) {
        t_vals[i] = near + i * (far - near) / (num_samples - 1);
    }
}

// Launcher for the CUDA kernel
torch::Tensor uniform_sampling_cuda(float near, float far, int num_samples) {
    auto t_vals = torch::zeros({num_samples}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    int threads_per_block = 256;
    int num_blocks = (num_samples + threads_per_block - 1) / threads_per_block;

    uniform_sampling_kernel<<<num_blocks, threads_per_block>>>(
        t_vals.data_ptr<float>(), near, far, num_samples);

    cudaDeviceSynchronize();

    return t_vals;
}

// Helper function to search CDF
__device__ int search_cdf(float *cdf, float u, int num_samples) {
    int low = 0, high = num_samples - 1;
    while (low < high) {
        int mid = (low + high) / 2;
        if (cdf[mid] < u) low = mid + 1;
        else high = mid;
    }
    return low;
}

// CUDA Kernel for Hierarchical Sampling
__global__ void hierarchical_sampling_kernel(float *coarse_samples, float *cdf, float *fine_samples, int num_fine_samples, int num_coarse_samples) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_fine_samples) {
        curandState state;
        curand_init(1234 + i, i, 0, &state);
        float u = curand_uniform(&state);
        int idx = search_cdf(cdf, u, num_coarse_samples);
        fine_samples[i] = coarse_samples[idx];
    }
}

// Launcher for the CUDA kernel
torch::Tensor hierarchical_sampling_cuda(torch::Tensor coarse_samples, torch::Tensor weights, int num_fine_samples) {
    int num_coarse_samples = coarse_samples.size(0);
    auto cdf = torch::cumsum(weights, 0) / weights.sum();
    auto fine_samples = torch::zeros({num_fine_samples}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    int threads_per_block = 256;
    int num_blocks = (num_fine_samples + threads_per_block - 1) / threads_per_block;

    hierarchical_sampling_kernel<<<num_blocks, threads_per_block>>>(
        coarse_samples.data_ptr<float>(), cdf.data_ptr<float>(), fine_samples.data_ptr<float>(), num_fine_samples, num_coarse_samples);

    cudaDeviceSynchronize();

    return fine_samples;
}

// CUDA Kernel for Inverse Transform Sampling
__global__ void inverse_transform_sampling_kernel(float* cdf, int* samples, int num_samples, int cdf_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_samples) {
        curandState state;
        curand_init(1234, i, 0, &state);
        float u = curand_uniform(&state);
        int idx = search_cdf(cdf, u, cdf_size);
        samples[i] = idx;
    }
}

// Launcher for the CUDA kernel
torch::Tensor inverse_transform_sampling_cuda(torch::Tensor weights, int num_samples) {
    int cdf_size = weights.size(0);
    auto cdf = torch::cumsum(weights, 0) / weights.sum();
    auto samples = torch::zeros({num_samples}, torch::device(torch::kCUDA).dtype(torch::kInt32));

    int threads_per_block = 256;
    int num_blocks = (num_samples + threads_per_block - 1) / threads_per_block;

    inverse_transform_sampling_kernel<<<num_blocks, threads_per_block>>>(
        cdf.data_ptr<float>(), samples.data_ptr<int>(), num_samples, cdf_size);

    cudaDeviceSynchronize();

    return samples;
}