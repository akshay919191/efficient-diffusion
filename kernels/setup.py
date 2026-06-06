# import os
# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# extra_compile_args = {
#     'cxx': ['-O3'],
#     'nvcc': [
#         '-O3',
#         '--use_fast_math',
#     ]
# }

# setup(
#     name='custom_flash_attn',
#     ext_modules=[
#         CUDAExtension(
#             name='custom_flash_attn',
#             sources=['flashattn/flashattn.cu'], 
#             extra_compile_args=extra_compile_args
#         ),
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     }
# )

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

include_dir = os.path.abspath("welford")

setup(
    name='welford_cuda_ext',
    ext_modules=[
        CUDAExtension(
            name='welford_cuda_ext',
            sources=[
                'welford/extension.cpp',
                'welford/welford.cu'        # <-- Updated from welford_kernel.cu to welford.cu
            ],
            include_dirs=[include_dir],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-arch=sm_80']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)