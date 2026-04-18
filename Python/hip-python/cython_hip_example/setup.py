import os
import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")

setup(
    ext_modules=cythonize([
        Extension(
            "matrix_prep",
            sources=["matrix_prep.pyx"],
            include_dirs=[np.get_include()],
            library_dirs=[os.path.join(ROCM_PATH, "lib")],
            libraries=["amdhip64"],
            extra_compile_args=["-D__HIP_PLATFORM_AMD__"],
        )
    ], compiler_directives={"language_level": 3})
)
