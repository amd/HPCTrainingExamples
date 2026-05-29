from setuptools import Extension, setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        [Extension("array_sum", ["array_sum.pyx"])],
        compiler_directives={"language_level": 3},
    )
)
