# Anyfold

C/C++ library to provide a unified API for high-performance convolutions.

## How to Build

### Dependencies

* boost (unit test framework, multi-array)
* cmake (to build it)
* c/c++ compiler (notably gcc)
* OpenCL (optional)

### CLI

```bash
$ git clone https://github.com/stebecker/anyfold
$ cd anyfold
$ mkdir build
$ cd build
$ cmake ..
$ make
$ ctest
```

The following cmake flags are supported:
* ```CMAKE_INSTALL_PREFIX``` to provide a custom installation directory
* ```BUILD_OPENCL_ANYFOLD``` to build anyfold with OpenCL support

## target platforms

As this is an educational project (until stable), we target Linux primarily using regular x86 instructions. The ultimate goal is to provide all functionality based on OpenCL (and potentially CUDA).

## usage

* CPU functionality is provided in namespace ```anyfold::cpu```
* OpenCL functionality is provided in namespace ```anyfold::opencl```
