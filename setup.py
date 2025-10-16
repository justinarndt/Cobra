# setup.py

import os
import sys
import platform
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the C++ extension.")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # The full path to the package source directory. This is where the
        # compiled C++ extension MUST be placed.
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}'
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += [f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}']
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
            build_args += ['--', '-j2']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='cobra',
    version='0.1.0',
    author='Justin Arndt',
    author_email='justinarndtai@gmail.com',
    description='A unified architecture for accelerated Python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    # This correctly declares that 'cobra_core' is a module INSIDE the 'cobra' package.
    ext_modules=[CMakeExtension('cobra.cobra_core')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
    python_requires='>=3.6',
)

