from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

Extension = Extension(
    name = "m",
    sources = ["matrix.pyx", "matriz.c"]
)

setup(
    ext_modules= cythonize(Extension, language_level=3)
)