"""Extend setuptools built_ext command to compile CUDA device code in case of
separate compilation."""

import os
import sys

from setuptools.command.build_ext import build_ext as _sut_build_ext
from setuptools.extension import Library
from setuptools import Extension as _sut_Extension

from distutils.command.build_ext import build_ext as _du_build_ext
from distutils.dep_util import newer_group
from distutils.spawn import spawn
from distutils import log

from string import Template

from .config import *

# CUDA architechture template
cu_arch_template = Template("--generate-code=arch=compute_${arch},"
                            "code=[compute_${arch},sm_${arch}]")

# overwrite method build_extension of setuptools.build_ext
class build_ext(_sut_build_ext):
    def build_extension(self, ext):
        ext._convert_pyx_sources_to_lang()
        _compiler = self.compiler
        try:
            if isinstance(ext, Library):
                self.compiler = self.shlib_compiler

            # call custom_du_build_ext
            # instead of calling build_ext from distutils
            custom_du_build_ext.build_extension(self, ext)

            if ext._needs_stub:
                build_lib = self.get_finalized_command('build_py').build_lib
                self.write_stub(build_lib, ext)
        finally:
            self.compiler = _compiler


# overwrite build_ext from distutils
class custom_du_build_ext(_du_build_ext):
    def build_extension(self, ext):
        # code copied from distutils/command/build_ext.py
        sources = ext.sources
        if sources is None or not isinstance(sources, (list, tuple)):
            raise DistutilsSetupError(
                "in 'ext_modules' option (extension '%s'), "
                "'sources' must be present and must be "
                "a list of source filenames" % ext.name
            )
        sources = sorted(sources)
        ext_path = self.get_ext_fullpath(ext.name)
        depends = sources + ext.depends
        if not (self.force or newer_group(depends, ext_path, 'newer')):
            log.debug("skipping '%s' extension (up-to-date)", ext.name)
            return
        else:
            log.info("building '%s' extension", ext.name)
        sources = self.swig_sources(sources, ext)
        extra_args = ext.extra_compile_args or []
        macros = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))
        # compile step
        if MERLIN_DEBUG:
            self.debug = True
        objects = self.compiler.compile(
            sources,
            output_dir=self.build_temp,
            macros=macros,
            include_dirs=ext.include_dirs,
            debug=self.debug,
            extra_postargs=extra_args,
            depends=ext.depends,
        )

        # cuda device linker
        if MERLIN_CUDA:
            obj_dir, obj_fname = os.path.split(objects[0])
            device_obj_fname = "cu_dev_linker_" + obj_fname
            device_linker = os.path.join(obj_dir, device_obj_fname)
            arch_args = [cu_arch_template.substitute(arch=str(arch))
                         for arch in CUDA_ARCHITECHTURE]
            dlink_option = ["-forward-unknown-to-host-compiler",
                            "-Wno-deprecated-gpu-targets",
                            "-shared", "-dlink", "-dlto"]
            dlink_option += arch_args
            lib_dlink = []
            if sys.platform == "win32":
                from distutils.ccompiler import gen_lib_options
                # dlink option
                dlink_option += ["-Xcompiler=\"/EHsc\"", "-Xcompiler=\"/MD\"",
                                 "-Xcompiler=\"/Ob2\"", "-Xcompiler=\"/O2\"",
                                 "-D_WINDOWS", "-DNDEBUG"]
                dlink_option += [
                    f"-LIBPATH:\"{os.path.dirname(CUDALIB)}\"",
                    f"-LIBPATH:\"{MERLIN_BIN_DIR}\""
                ]
                # system paths
                temp = [self.get_libraries(ext), ext.library_dirs,
                        ext.runtime_library_dirs]
                fixed_args = self.compiler._fix_lib_args(*temp)
                libraries, library_dirs, runtime_library_dirs = fixed_args
                sys_libpath = gen_lib_options(self.compiler, library_dirs,
                                              runtime_library_dirs, libraries)
                dlink_option += [p.replace("/LIBPATH:", "-LIBPATH:\"") + "\""
                                 for p in sys_libpath
                                 if p.startswith("/LIBPATH:")]
                # linked library to dlink
                if MERLIN_LIBKIND == "SHARED":
                    lib_dlink = ["merlinglobal.lib", "merlincuda.lib"]
                else:
                    lib_dlink = ["merlin.lib"]
                lib_dlink += [CUDADRIVER, CUDART, CUDADEVRT]
            elif sys.platform == "linux":
                # dlink option
                dlink_option += ["-O3", "-DNDEBUG", "-Xcompiler=-fPIC"]
                dlink_option += [f"-L\"{os.path.dirname(CUDALIB)}\""]
                # linked library to dlink
                dlink_option += [f"-L\"{MERLIN_BIN_DIR}\""]
                if MERLIN_LIBKIND == "SHARED":
                    lib_dlink = ["-lmerlinglobal", "-lmerlincuda"]
                else:
                    lib_dlink = ["-lmerlin"]
                lib_dlink += ["-lcuda", "-lcudart_static", "-lcudadevrt"]
            self.spawn([NVCC] + dlink_option + objects
                       + ["-o", device_linker] + lib_dlink)
            objects += [device_linker]

        # link step
        self._built_objects = objects[:]
        if ext.extra_objects:
            objects.extend(ext.extra_objects)
        extra_args = ext.extra_link_args or []
        language = ext.language or self.compiler.detect_language(sources)
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
