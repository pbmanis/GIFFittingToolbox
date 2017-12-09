#python setup.py build_ext --inplace
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

# setup(
#     ext_modules=cythonize("gif_mcurrent.pyx", extra_compile_args = ["-ffast-math"]),
#     include_dirs=[numpy.get_include()]
# )
#


ext_modules=[ Extension("gif_mcurrent",
              ["gif_mcurrent.pyx"],
              libraries=["m"],
              extra_compile_args = ["-ffast-math"])]

setup(
  name = "gif_mcurrent",
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules)
  