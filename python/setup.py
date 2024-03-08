from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from distutils.command.clean import clean
from setuptools.command.build_py import build_py
import os
from pathlib import Path
import shutil
import sysconfig
import sys
import subprocess


# Taken from https://github.com/pytorch/pytorch/blob/master/tools/setup_helpers/env.py
def check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def get_cmake_project_source_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def get_cmake_project_binary_dir():
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    cmake_dir = Path(get_cmake_project_source_dir()) / "build" / dir_name
    return cmake_dir


def get_cmake_build_type():
    return os.getenv('INDUCTOR_MLIR_BUILD_TYPE', 'Release')


def get_llvm_dir():
    return os.getenv('INDUCTOR_MLIR_LLVM_DIR', None)


class CMakeClean(clean):

    def initialize_options(self):
        clean.initialize_options(self)
        self.build_temp = get_cmake_project_binary_dir()


class CMakeBuildPy(build_py):

    def run(self) -> None:
        self.run_command('build_ext')
        return super().run()


class CMakeExtension(Extension):

    def __init__(self, name, path):
        super().__init__(name, [])
        self.path = path


class CMakeBuildExt(build_ext):

    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        cmake_source_dir = get_cmake_project_source_dir()
        cmake_build_dir = get_cmake_project_binary_dir()
        cmake_build_dir.mkdir(parents=True, exist_ok=True)
        cmake_build_type = get_cmake_build_type()

        cmake_library_output_dir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.path)))

        build_args = []
        cmake_args = [
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DCMAKE_BUILD_TYPE=" + cmake_build_type,
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + cmake_library_output_dir,
        ]

        if shutil.which('ninja'):
            cmake_args += ['-G', 'Ninja']

        llvm_dir = get_llvm_dir()
        if llvm_dir is not None:
            cmake_args += ['-DLLVM_DIR=' + llvm_dir]

        if check_env_flag("INDUCTOR_MLIR_BUILD_WITH_CLANG_LLD", "ON"):
            cmake_args += [
                "-DCMAKE_C_COMPILER=clang",
                "-DCMAKE_CXX_COMPILER=clang++",
                "-DCMAKE_LINKER=lld",
                "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
                "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
                "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
            ]

        if check_env_flag("INDUCTOR_MLIR_BUILD_WITH_CCACHE", "ON"):
            cmake_args += [
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
            ]

        subprocess.check_call(["cmake", cmake_source_dir] + cmake_args,
                              cwd=cmake_build_dir)
        subprocess.check_call(["cmake", "--build", "."] + build_args,
                              cwd=cmake_build_dir)


setup(ext_modules=[CMakeExtension("inductor_mlir", "inductor_mlir/_C/")],
      cmdclass={
          "build_ext": CMakeBuildExt,
          "build_py": CMakeBuildPy,
          "clean": CMakeClean,
      })
