from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Read the README for the long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='nerfboost',
    version='0.0.5',
    description='A CUDA-enhanced package for common NeRF model operations',
    long_description=long_description,  # This will pull the model description from the README
    long_description_content_type="text/markdown",  # Ensure markdown is correctly interpreted
    author='Akshay Pappu',
    license='MIT License (MIT)',
    python_requires='>=3.8',
    install_requires=[
        'torch',  
    ],
    ext_modules=[
        CUDAExtension(
            name='nerfboost',
            sources=[
                'nerfboost/positional_encoding.cu', 
                'nerfboost/sampling.cu',
                'nerfboost/volume_rendering.cu',
                'nerfboost/mlp_network.cu',
                'nerfboost/rendering_loss.cu',
                'nerfboost/generate_rays.cu',
                'nerfboost/pybind_module.cu'
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
