from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "compute",
        sources=["compute.pyx", "hpc_kernels.c"],
        include_dirs=[np.get_include(), "."],
    )
]

setup(
    name="hpc-cython-kernels",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
