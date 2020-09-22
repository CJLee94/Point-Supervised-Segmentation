from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='test app',
    ext_modules=cythonize("_watershed_cy.pyx"),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)
