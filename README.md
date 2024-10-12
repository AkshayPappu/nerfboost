# Nerfboost - CUDA-accelerated NeRF Model Training for AR/VR Experiences

nerfboost is a CUDA-accelerated Python package designed to make Neural Radiance Fields (NeRF) model training easier and more efficient, particularly for applications in augmented reality (AR) and virtual reality (VR). NeRF models are essential for generating high-quality, 3D scene reconstructions from 2D images or views, making them perfect for AR/VR experiences. nerfboost provides high-performance implementations of key NeRF operations, accelerating training and rendering times for AR/VR projects.

## Features

- **CUDA-accelerated functions** for efficient neural rendering tasks.
- Implements key operations for NeRF models like:
  - Positional encoding
  - Stratified, uniform, hierarchical, and inverse sampling
  - Volume rendering
  - MLP network processing
  - Rendering loss computation
  - Ray generation
- Easy integration with PyTorch using custom CUDA kernels.

## Prerequisites

Ensure PyTorch is installed. 

You can install **Torch** using the following command:

```bash
pip install torch
```

## Installation

You can install **nerfboost** using the following command:

```bash
pip install nerfboost
```

## Usage

# Initialization

```
import nerfboost
```

# Positional encoding
The positional encoding transforms input coordinates into a higher-dimensional space, crucial for NeRF models.

```
x = torch.randn(10, 3)  # Input coordinates (e.g., 10 points in 3D space)
L = 10  # Number of encoding frequencies
encoded_positions = nerfboost.positional_encoding_cuda(x, L)

```

# Stratified and Uniform Sampling
Stratified sampling generates sample points along a ray between near and far planes:
Uniform sampling generates evenly spaced samples between near and far planes:


```
near = 0.1
far = 4.0
num_samples = 64
stratified_samples = nerfboost.stratified_sampling_cuda(near, far, num_samples)
unifrom_samples = nerfboost.uniform_sampling_cuda(near, far, num_samples)
```

# Hierarchical Sampling
Refines samples based on coarse predictions:

```
coarse_samples = torch.randn(num_samples)  # Coarse samples from the network
weights = torch.randn(num_samples)  # Corresponding weights
num_fine_samples = 128
fine_samples = nerfboost.hierarchical_sampling_cuda(coarse_samples, weights, num_fine_samples)
```

# Inverse Transform Sampling
Generates samples using inverse transform sampling based on the given probability distribution:

```
weights = torch.rand(64)  # Probability distribution
num_samples = 128
samples = nerfboost.inverse_transform_sampling_cuda(weights, num_samples)
```

# Volume Rendering
Computes the final color and density for each ray based on input densities, colors, and distances:

```
densities = torch.rand(64)  # Density for each sample
colors = torch.rand(64, 3)  # RGB color for each sample
distances = torch.rand(64)  # Distance along the ray
final_colors = nerfboost.volume_rendering_cuda(densities, colors, distances)
```

# MLP Network Processing
Processes sampled points through a multi-layer perceptron (MLP) network:

```
sampled_points = torch.randn(64, 3)  # Points sampled along the rays
directions = torch.randn(64, 3)  # Ray directions
weights = [torch.randn(3, 64), torch.randn(64, 64), torch.randn(64, 3)]  # MLP weights
output = nerfboost.mlp_network_cuda(sampled_points, directions, weights)
```

# Rendering Loss (MSE)
Computes the mean squared error (MSE) between the rendered image and the ground truth image:

```
rendered_image = torch.rand(256, 256, 3)  # Rendered output
ground_truth_image = torch.rand(256, 256, 3)  # Ground truth image
loss = nerfboost.rendering_loss_cuda(rendered_image, ground_truth_image)
```

# Ray Generation
Generates rays for rendering using the camera intrinsics and resolution:

```
camera_intrinsics = torch.rand(3, 3)  # Camera intrinsic matrix
H = 256  # Image height
W = 256  # Image width
origins, directions = nerfboost.generate_rays_cuda(camera_intrinsics, H, W)
```

## License

This project is licensed under the MIT License.

## Author

**Akshay Pappu**