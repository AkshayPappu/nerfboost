#include <torch/extension.h>

// CUDA Kernels for the custom functions
torch::Tensor positional_encoding_cuda(torch::Tensor x, int L);
torch::Tensor stratified_sampling_cuda(float near, float far, int num_samples);
torch::Tensor uniform_sampling_cuda(float near, float far, int num_samples);
torch::Tensor hierarchical_sampling_cuda(torch::Tensor coarse_samples, torch::Tensor weights, int num_fine_samples);
torch::Tensor inverse_transform_sampling_cuda(torch::Tensor weights, int num_samples);
torch::Tensor volume_rendering_cuda(torch::Tensor densities, torch::Tensor colors, torch::Tensor distances);
torch::Tensor mlp_network_cuda(torch::Tensor sampled_points, torch::Tensor directions, std::vector<torch::Tensor> weights);
torch::Tensor rendering_loss_cuda(torch::Tensor rendered_image, torch::Tensor ground_truth_image);
std::tuple<torch::Tensor, torch::Tensor> generate_rays_cuda(torch::Tensor camera_intrinsics, int H, int W);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stratified_sampling_cuda", &stratified_sampling_cuda, "Stratified Sampling CUDA");
    m.def("positional_encoding_cuda", &positional_encoding_cuda, "Positional Encoding CUDA");
    m.def("uniform_sampling_cuda", &uniform_sampling_cuda, "Uniform Sampling CUDA");
    m.def("hierarchical_sampling_cuda", &hierarchical_sampling_cuda, "Hierarchical Sampling CUDA");
    m.def("inverse_transform_sampling_cuda", &inverse_transform_sampling_cuda, "Inverse Transform Sampling CUDA");
    m.def("volume_rendering_cuda", &volume_rendering_cuda, "Volume Rendering CUDA");
    m.def("mlp_network_cuda", &mlp_network_cuda, "MLP Network using CUDA");
    m.def("rendering_loss_cuda", &rendering_loss_cuda, "MSE loss for rendering using CUDA");
    m.def("generate_rays_cuda", &generate_rays_cuda, "Generate rays using CUDA");
}
