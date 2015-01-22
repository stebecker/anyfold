add_library(anyfold opencl/convolution3DCLBuffer.cpp opencl/convolution3DCLBufferLocalMem.cpp opencl/convolution3DCLImage.cpp opencl/convolution3DCLImageLocalMem.cpp)
target_link_libraries(anyfold ${OpenCL_LIBRARIES})
set_target_properties(anyfold PROPERTIES COMPILE_FLAGS "-I${PROJECT_SOURCE_DIR}/include")
