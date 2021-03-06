#ifndef CONVOLUTION3DCLBUFFER_HPP
#define CONVOLUTION3DCLBUFFER_HPP

#include <vector>
#include <string>

#ifdef __APPLE__
	#include "Opencl/opencl.hpp"
#else
	#include "CL/cl.hpp"
#endif

#include "image_stack_utils.h"

namespace anyfold {

namespace opencl {

class Convolution3DCLBuffer
{
public:
	Convolution3DCLBuffer() = default;
	~Convolution3DCLBuffer() = default;

	bool setupCLcontext();
	void createProgramAndLoadKernel(const std::string& fileName,
	                                const std::string& kernelName,
	                                size_t const* filterSize);
	void setupKernelArgs(image_stack_cref _image,
	                     image_stack_cref _kernel,
	                     const std::vector<int>& _offset);
	void execute();
	void getResult(image_stack_ref result);
	// void convolve3D(/* something image3D, something filterkernel3D */);


private:
	void createProgram(const std::string& source,size_t const* filterSize);
	void loadKernel(const std::string& kernelName);
	std::string getDeviceInfo(cl::Device device, cl_device_info info);
	std::string getDeviceName(cl::Device device);
	std::string getPlatformInfo(cl::Platform platform, cl_platform_info info);

	void checkError(cl_int status, const char* label,
	                const char* file, int line);


private:
	cl::Context context;
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> devices;

	cl::Program program;
	cl::Kernel kernel;
	cl::CommandQueue queue;

	cl_int status = CL_SUCCESS;

	cl::Buffer inputBuffer;
	cl::Buffer outputBuffer;
	cl::Buffer filterWeightsBuffer;
	std::size_t imageSize[3];
	std::size_t imageSizeInner[3];
	std::size_t filterSize[3];
};

} /* namespace opencl */
} /* namespace anyfold */

#endif /* CONVOLUTION3DCLBUFFER_HPP */
