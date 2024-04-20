"""Extend setuptools built_ext command to compile CUDA device code in case of
separate compilation."""

import os
import re
import sys
from string import Template
from threading import Thread
from typing import Any, Dict, List

from pybind11.setup_helpers import Pybind11Extension

from setuptools.command.build_ext import build_ext as _sut_build_ext
from setuptools.extension import Library

from distutils import log
from distutils.ccompiler import gen_lib_options
from distutils.command.build_ext import build_ext as _du_build_ext
from distutils.dep_util import newer_group
from distutils.errors import DistutilsSetupError

# ----------------------------------------------------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------------------------------------------------

# CUDA architecture template
cu_arch_template = Template("--generate-code=arch=compute_${arch}," "code=[compute_${arch},sm_${arch}]")


# thread object with result
class ResultedThread(Thread):
    def run(self):
        self.result = self._target(*self._args, **self._kwargs)


# parallel compile source
def parallel_compile(sources, compiler, compile_opt, numthreads=os.cpu_count()):
    objects = []
    num_pass = len(sources) // numthreads
    # compile chunks
    for p in range(num_pass):
        compile_threads = []
        for i_src in range(p * numthreads, (p + 1) * numthreads):
            source = sources[i_src]
            compile_thread = ResultedThread(target=compiler.compile, args=[[source]], kwargs=compile_opt)
            compile_thread.start()
            compile_threads.append(compile_thread)
        for thread in compile_threads:
            thread.join()
            objects.extend(thread.result)
    # compile remainder
    compile_threads = []
    for i_src in range(num_pass * numthreads, len(sources)):
        source = sources[i_src]
        compile_thread = ResultedThread(target=compiler.compile, args=[[source]], kwargs=compile_opt)
        compile_thread.start()
        compile_threads.append(compile_thread)
    for thread in compile_threads:
        thread.join()
        objects.extend(thread.result)
    return objects


# ----------------------------------------------------------------------------------------------------------------------
# Overwrite Pybind11 Extension
# ----------------------------------------------------------------------------------------------------------------------


def check_and_throw(key: str, kwargs: Dict[str, Any]):
    if key not in kwargs:
        message = f"Expected argument {key} provided in CUDA mode"
        raise ValueError(message)
    return kwargs.pop(key)


class Extension(Pybind11Extension):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # add CUDA features to the extension
        self.use_cuda = kwargs.pop("cuda")
        if self.use_cuda:
            self.nvcc_executable = check_and_throw("nvcc_executable", kwargs)
            self.cuda_arch = check_and_throw("cuda_arch", kwargs)
            self.cuda_linkdir = check_and_throw("cuda_linkdir", kwargs)
            self.lib_cudart = check_and_throw("lib_cudart", kwargs)
            self.lib_cudadevrt = check_and_throw("lib_cudadevrt", kwargs)
            self.lib_cudadriver = check_and_throw("lib_cudadriver", kwargs)
            self.libs_device_linker = check_and_throw("libs_device_linker", kwargs)
        # constructor for parent class
        super().__init__(*args, **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
# Add CUDA compile step
# ----------------------------------------------------------------------------------------------------------------------


# CUDA device linker step
def cuda_device_linker(self: _du_build_ext, ext: Extension, objects: List[str]):
    # get name of created object for device linker step
    obj_dir = os.path.dirname(objects[0])
    device_obj_fname = f"cu_dev_linker_{ext.name.replace('.', '_')}"
    obj_device_linker = os.path.join(obj_dir, device_obj_fname)
    # get generic options for device linker step
    arch_args = [cu_arch_template.substitute(arch=str(arch)) for arch in ext.cuda_arch]
    dlink_option = [
        "-forward-unknown-to-host-compiler",
        "-Wno-deprecated-gpu-targets",
        "-shared",
        "-dlink",
        "-dlto",
    ]
    dlink_option += arch_args
    # get list of libraries for linking
    lib_dlink = ext.libs_device_linker + [ext.lib_cudart, ext.lib_cudadevrt, ext.lib_cudadriver]
    # platform specific options
    if sys.platform == "win32":
        # dlink option
        dlink_option += [
            '-Xcompiler="/EHsc"',
            '-Xcompiler="/MD"',
            '-Xcompiler="/Ob2"',
            '-Xcompiler="/O2"',
            "-D_WINDOWS",
            "-DNDEBUG",
        ]
        dlink_option += [f'-LIBPATH:"{lib}"' for lib in ext.cuda_linkdir]
        # system paths
        temp = [self.get_libraries(ext), ext.library_dirs, ext.runtime_library_dirs]
        fixed_args = self.compiler._fix_lib_args(*temp)
        libraries, library_dirs, runtime_library_dirs = fixed_args
        sys_libpath = gen_lib_options(self.compiler, library_dirs, runtime_library_dirs, libraries)
        dlink_option += [p.replace("/LIBPATH:", '-LIBPATH:"') + '"' for p in sys_libpath if p.startswith("/LIBPATH:")]
        obj_device_linker += ".obj"
    elif sys.platform == "linux":
        # dlink option
        dlink_option += ["-O3", "-DNDEBUG", "-Xcompiler=-fPIC"]
        dlink_option += [f'-L"{lib}"' for lib in ext.cuda_linkdir]
        # transform to libname
        pattern = re.compile(r"lib([\w_]+)\.[a|so]")
        for i_lib, lib in enumerate(lib_dlink):
            lib_dlink[i_lib] = f"-l{pattern.match(lib).group(1)}"
        obj_device_linker += ".o"
    # execute
    self.spawn([ext.nvcc_executable] + dlink_option + objects + ["-o", obj_device_linker] + lib_dlink)
    objects += [obj_device_linker]
    return objects


# overwrite method build_extension of setuptools.build_ext
class BuildExt(_sut_build_ext):
    def build_extension(self, ext):
        ext._convert_pyx_sources_to_lang()
        _compiler = self.compiler
        try:
            if isinstance(ext, Library):
                self.compiler = self.shlib_compiler

            # call custom_du_build_ext instead of calling build_ext from distutils
            CustomBuildExt.build_extension(self, ext)

            if ext._needs_stub:
                build_lib = self.get_finalized_command("build_py").build_lib
                self.write_stub(build_lib, ext)
        finally:
            self.compiler = _compiler


# overwrite build_ext from distutils
class CustomBuildExt(_du_build_ext):
    def build_extension(self, ext: Extension):
        # code copied from distutils/command/build_ext.py
        sources = ext.sources
        if sources is None or not isinstance(sources, (list, tuple)):
            msg = (
                f"in 'ext_modules' option (extension {ext.name}), 'sources' must be present and "
                f"must be a list of source filenames"
            )
            raise DistutilsSetupError(msg)
        sources = sorted(sources)
        ext_path = self.get_ext_fullpath(ext.name)
        depends = sources + ext.depends
        if not (self.force or newer_group(depends, ext_path, "newer")):
            log.debug("skipping '%s' extension (up-to-date)", ext.name)
            return
        else:
            log.info("building '%s' extension", ext.name)
        sources = self.swig_sources(sources, ext)
        extra_args = ext.extra_compile_args or []
        macros = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))

        # disable parallel compilation in debug mode
        if not (self.parallel):
            self.parallel = os.cpu_count()
        if self.debug:
            self.parallel = 1

        # compile step
        compile_opt = {
            "output_dir": self.build_temp,
            "macros": macros,
            "include_dirs": ext.include_dirs,
            "debug": self.debug,
            "extra_postargs": extra_args,
            "depends": ext.depends,
        }
        objects = parallel_compile(sources, self.compiler, compile_opt, self.parallel)

        # cuda device linker step
        if ext.use_cuda:
            objects = cuda_device_linker(self, ext, objects)

        # link step
        self._built_objects = objects[:]
        if ext.extra_objects:
            objects.extend(ext.extra_objects)
        extra_args = ext.extra_link_args or []
        language = ext.language or self.compiler.detect_language(sources)
        name = ext.name.split(".")
        if name[-1] == "__init__":
            if len(name) < 2:
                raise ValueError("Extension with name __init__ must be acquired by parent package name.")
            ext.name = ".".join(name[:-1])
        self.compiler.link_shared_object(
            objects,
            ext_path,
            libraries=self.get_libraries(ext),
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=extra_args,
            export_symbols=self.get_export_symbols(ext),
            debug=self.debug,
            build_temp=self.build_temp,
            target_lang=language,
        )
        ext.name = ".".join(name)
