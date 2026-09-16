from pathlib import Path
import os

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


PODEM_DIR = Path(__file__).resolve().parent
os.chdir(PODEM_DIR)


CORE_SOURCES = [
    "atpg.cpp",
    "input.cpp",
    "level.cpp",
    "sim.cpp",
    "podem.cpp",
    "saf_compaction.cpp",
    "init_flist.cpp",
    "faultsim.cpp",
    "tdfsim.cpp",
    "tdfatpg.cpp",
    "display.cpp",
    "python_bindings.cpp",
]


class BuildExt(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type == "msvc":
            compile_args = ["/std:c++14", "/EHsc", "/O2"]
            link_args = ["/MANIFEST:NO"]
        else:
            compile_args = ["-std=c++11", "-O3"]
            link_args = []
        for extension in self.extensions:
            extension.extra_compile_args = compile_args
            extension.extra_link_args = link_args
        super().build_extensions()


pybind11_include = PODEM_DIR.parent / "third_party" / "pybind11" / "include"
if not (pybind11_include / "pybind11" / "pybind11.h").is_file():
    raise RuntimeError("vendored pybind11 headers are missing: " + str(pybind11_include))


extension = Extension(
    "cpp_podem",
    [str(PODEM_DIR / "src" / source) for source in CORE_SOURCES],
    include_dirs=[str(PODEM_DIR / "src"), str(pybind11_include)],
    language="c++",
)


setup(
    name="cpp-podem-catalog",
    version="0.1.0",
    description="PODEM stuck-at fault catalog and ordered ATPG bridge",
    python_requires=">=3.9",
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExt},
)
