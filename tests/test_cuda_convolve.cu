#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE CUDA_CONVOLUTION
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>
#include <vector>
#include <functional>
//#include "padd_utils.h"
#include "cuda_convolve.cuh"
#include "cuda_helpers.cuh"


BOOST_AUTO_TEST_CASE( trivial_convolve )
{

  using namespace cuda_convolve;

  static const unsigned kernel_axis_length = 3;
  static const unsigned image_axis_length = 64+2*(kernel_axis_length); 
  static const unsigned num_pixels = image_axis_length*image_axis_length*image_axis_length;
  static const unsigned num_k_pixels = kernel_axis_length*kernel_axis_length*kernel_axis_length;

  std::vector<float> output(num_pixels);
  std::vector<float> input (num_pixels);
  std::vector<float> kernel(num_k_pixels);

  std::fill(kernel.begin(), kernel.end(),0.f);
  std::fill(input.begin(),input.end(),10); 
  
  unsigned long checksum = num_pixels*10;

  float* d_input = 0;
  float* d_kernel = 0;
  float* d_output = 0;
  
  //allocat ememory in GPU
  HANDLE_ERROR( cudaMalloc( (void**)&(d_input), num_pixels*sizeof(float) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&(d_output), num_pixels*sizeof(float) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&(d_kernel), (num_k_pixels)*sizeof(float) ) );

  HANDLE_ERROR( cudaMemcpy( d_input, &input[0] , num_pixels*sizeof(float) , cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( d_output, &output[0] , num_pixels*sizeof(float) , cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( d_kernel, &kernel[0] , num_k_pixels*sizeof(float) , cudaMemcpyHostToDevice ) );

  dim3 threads(128);
  dim3 blocks((num_pixels + threads.x -1)/threads.x);

  uint3 padded_image_dims;
  padded_image_dims.x  = image_axis_length;
  padded_image_dims.y  = image_axis_length;
  padded_image_dims.z  = image_axis_length;

  static_convolve<3u><<<threads,blocks>>>(d_input, d_kernel, d_output, padded_image_dims);

  HANDLE_ERROR( cudaMemcpy( &output[0], d_output , output.size()*sizeof(float) , cudaMemcpyDeviceToHost ) );

  unsigned long sum = std::accumulate(output.begin(),output.end(),0.f);
  BOOST_CHECK_LT(sum, checksum);

  HANDLE_ERROR( cudaFree( d_input ) );
  HANDLE_ERROR( cudaFree( d_kernel ) );
  HANDLE_ERROR( cudaFree( d_output ) );

}

// BOOST_AUTO_TEST_CASE( identity_convolve )
// {
  
//   using namespace cuda_convolve;

//   float sum_expected = std::accumulate(image_.data(), image_.data() + image_.num_elements(),0.f);

//   zero_padd<image_stack> padder(image_.shape(), identity_kernel_.shape());
//   image_stack padded_image(padder.extents_, image_.storage_order());
//   padder.insert_at_offsets(image_, padded_image);
  
//   std::vector<int> extents_as_int(padder.extents_.size());
//   std::copy(padder.extents_.begin(), padder.extents_.end(), extents_as_int.begin());

//   convolution3DfftCUDAInPlace(padded_image.data(), &extents_as_int[0], 
//   			  identity_kernel_.data(),&kernel_dims_[0],
//   			  selectDeviceWithHighestComputeCapability());

//   float sum = std::accumulate(image_.data(), image_.data() + image_.num_elements(),0.f);
//   BOOST_CHECK_CLOSE(sum, sum_expected, .00001);


// }

// BOOST_AUTO_TEST_CASE( horizontal_convolve )
// {
//   using namespace cuda_convolve;

//   float sum_expected = std::accumulate(image_folded_by_horizontal_.data(), image_folded_by_horizontal_.data() + image_folded_by_horizontal_.num_elements(),0.f);

//   zero_padd<image_stack> padder(image_.shape(), horizont_kernel_.shape());
//   image_stack padded_image(padder.extents_, image_.storage_order());

//   padder.insert_at_offsets(image_, padded_image);
  
//   std::vector<int> extents_as_int(padder.extents_.size());
//   std::copy(padder.extents_.begin(), padder.extents_.end(), extents_as_int.begin());

//   convolution3DfftCUDAInPlace(padded_image.data(), &extents_as_int[0], 
// 			      horizont_kernel_.data(),&kernel_dims_[0],
// 			      selectDeviceWithHighestComputeCapability());

//   image_ = padded_image[ boost::indices[range(padder.offsets()[0], padder.offsets()[0]+image_dims_[0])][range(padder.offsets()[1], padder.offsets()[1]+image_dims_[1])][range(padder.offsets()[2], padder.offsets()[2]+image_dims_[2])] ];
  
//   float sum = std::accumulate(image_.data(), image_.data() + image_.num_elements(),0.f);

//     BOOST_REQUIRE_CLOSE(sum, sum_expected, .00001);
 
// }

// BOOST_AUTO_TEST_CASE( vertical_convolve )
// {
  
//   multiviewnative::zero_padd<multiviewnative::image_stack> padder(image_.shape(), vertical_kernel_.shape());
//   multiviewnative::image_stack padded_image(padder.extents_, image_.storage_order());

//   padder.insert_at_offsets(image_, padded_image);
  
//   std::vector<int> extents_as_int(padder.extents_.size());
//   std::copy(padder.extents_.begin(), padder.extents_.end(), extents_as_int.begin());

//   convolution3DfftCUDAInPlace(padded_image.data(), &extents_as_int[0], 
// 			      vertical_kernel_.data(),&kernel_dims_[0],
// 			      selectDeviceWithHighestComputeCapability());


//   float sum_expected = std::accumulate(image_folded_by_vertical_.data(), image_folded_by_vertical_.data() + image_folded_by_vertical_.num_elements(),0.f);

//   image_ = padded_image[ boost::indices[multiviewnative::range(padder.offsets()[0], padder.offsets()[0]+image_dims_[0])][multiviewnative::range(padder.offsets()[1], padder.offsets()[1]+image_dims_[1])][multiviewnative::range(padder.offsets()[2], padder.offsets()[2]+image_dims_[2])] ];

//   float sum = std::accumulate(image_.data(), image_.data() + image_.num_elements(),0.f);
//   BOOST_CHECK_CLOSE(sum, sum_expected, .00001);


// }

// BOOST_AUTO_TEST_CASE( depth_convolve )
// {
  
//   multiviewnative::zero_padd<multiviewnative::image_stack> padder(image_.shape(), depth_kernel_.shape());
//   multiviewnative::image_stack padded_image(padder.extents_, image_.storage_order());

//   padder.insert_at_offsets(image_, padded_image);
  
//   std::vector<int> extents_as_int(padder.extents_.size());
//   std::copy(padder.extents_.begin(), padder.extents_.end(), extents_as_int.begin());

//   convolution3DfftCUDAInPlace(padded_image.data(), &extents_as_int[0], 
// 			      depth_kernel_.data(),&kernel_dims_[0],
// 			      selectDeviceWithHighestComputeCapability());


//   float sum_expected = std::accumulate(image_folded_by_depth_.data(), image_folded_by_depth_.data() + image_folded_by_depth_.num_elements(),0.f);

//   image_ = padded_image[ boost::indices[multiviewnative::range(padder.offsets()[0], padder.offsets()[0]+image_dims_[0])][multiviewnative::range(padder.offsets()[1], padder.offsets()[1]+image_dims_[1])][multiviewnative::range(padder.offsets()[2], padder.offsets()[2]+image_dims_[2])] ];

//   float sum = std::accumulate(image_.data(), image_.data() + image_.num_elements(),0.f);
//   BOOST_CHECK_CLOSE(sum, sum_expected, .00001);


// }

// BOOST_AUTO_TEST_CASE( all1_convolve )
// {
  
//   multiviewnative::zero_padd<multiviewnative::image_stack> padder(image_.shape(), all1_kernel_.shape());
//   multiviewnative::image_stack padded_image(padder.extents_, image_.storage_order());

//   padder.insert_at_offsets(image_, padded_image);
  
//   std::vector<int> extents_as_int(padder.extents_.size());
//   std::copy(padder.extents_.begin(), padder.extents_.end(), extents_as_int.begin());

//   convolution3DfftCUDAInPlace(padded_image.data(), &extents_as_int[0], 
// 			      all1_kernel_.data(),&kernel_dims_[0],
// 			      selectDeviceWithHighestComputeCapability());


//   float sum_expected = std::accumulate(image_folded_by_all1_.data(), image_folded_by_all1_.data() + image_folded_by_all1_.num_elements(),0.f);

//   image_ = padded_image[ boost::indices[multiviewnative::range(padder.offsets()[0], padder.offsets()[0]+image_dims_[0])][multiviewnative::range(padder.offsets()[1], padder.offsets()[1]+image_dims_[1])][multiviewnative::range(padder.offsets()[2], padder.offsets()[2]+image_dims_[2])] ];

//   float sum = std::accumulate(image_.data(), image_.data() + image_.num_elements(),0.f);
//   BOOST_CHECK_CLOSE(sum, sum_expected, .00001);


// }

// BOOST_AUTO_TEST_SUITE_END()

// BOOST_FIXTURE_TEST_SUITE( gpu_convolution_works, multiviewnative::default_3D_fixture )

// BOOST_AUTO_TEST_CASE( trivial_convolve_newapi )
// {
//   using namespace cuda_convolve;
  
//   float* kernel = new float[kernel_size_];
//   std::fill(kernel, kernel+kernel_size_,0.f);

//   image_stack expected = image_;
//   std::fill(expected.data(), expected.data() + expected.num_elements(),0.f);

//   inplace_gpu_convolution(image_.data(), &image_dims_[0], 
// 			  kernel,&kernel_dims_[0],
// 			  selectDeviceWithHighestComputeCapability());


//   float sum = std::accumulate(image_.data(), image_.data() + image_size_,0.f);
//   try{
//     BOOST_REQUIRE_CLOSE(sum, 0.f, .00001);
//   }
//   catch(...){
//     std::cout << "expected:\n" << expected << "\n"
// 	      << "received:\n" << image_ << "\n";
//   }

//   delete [] kernel;
// }

// BOOST_AUTO_TEST_CASE( identity_convolve_newapi )
// {
//   using namespace cuda_convolve;

  
//   image_stack expected = image_;
//   float sum_original = std::accumulate(image_.data(), image_.data() + image_.num_elements(),0.f);
//   inplace_gpu_convolution(image_.data(), &image_dims_[0], 
//   			  identity_kernel_.data(),&kernel_dims_[0],
//   			  selectDeviceWithHighestComputeCapability());

  

//   float sum = std::accumulate(image_.data(), image_.data() + image_.num_elements(),0.f);
//   try{
//     BOOST_REQUIRE_CLOSE(sum, sum_original, .00001);
//   }
//   catch(...){
//     std::cout << boost::unit_test::framework::current_test_case().p_name << "\n"
// 	      << "expected:\n" << expected << "\n"
// 	      << "received:\n" << image_ << "\n";
//   }


// }

// BOOST_AUTO_TEST_CASE( horizontal_convolve_newapi )
// {
//   using namespace cuda_convolve;

  

//   float sum_original = std::accumulate(image_folded_by_horizontal_.data(), image_folded_by_horizontal_.data() + image_.num_elements(),0.f);
//   inplace_gpu_convolution(image_.data(), &image_dims_[0], 
//   			  horizont_kernel_.data(),&kernel_dims_[0],
//   			  selectDeviceWithHighestComputeCapability());

  

//   float sum = std::accumulate(image_.data(), image_.data() + image_.num_elements(),0.f);
//   //BOOST_CHECK_CLOSE(sum, sum_original, .00001);
//   try{
//     BOOST_REQUIRE_CLOSE(sum, sum_original, .00001);
//   }
//   catch(...){
//     std::cout << boost::unit_test::framework::current_test_case().p_name << "\n"
// 	      << "expected:\n" << image_folded_by_horizontal_ << "\n"
// 	      << "received:\n" << image_ << "\n";
//   }

// }

// BOOST_AUTO_TEST_CASE( vertical_convolve_newapi )
// {
//   using namespace cuda_convolve;

  

//   float sum_original = std::accumulate(image_folded_by_vertical_.data(), image_folded_by_vertical_.data() + image_.num_elements(),0.f);
//   inplace_gpu_convolution(image_.data(), &image_dims_[0], 
//   			  vertical_kernel_.data(),&kernel_dims_[0],
//   			  selectDeviceWithHighestComputeCapability());

  

//   float sum = std::accumulate(image_.data(), image_.data() + image_.num_elements(),0.f);
//   // BOOST_CHECK_CLOSE(sum, sum_original, .00001);
//  try{
//     BOOST_REQUIRE_CLOSE(sum, sum_original, .00001);
//   }
//   catch(...){
//     std::cout << boost::unit_test::framework::current_test_case().p_name << "\n" 
// 	      << "expected:\n" << image_folded_by_vertical_ << "\n"
// 	      << "received:\n" << image_ << "\n";
//   }

// }

// BOOST_AUTO_TEST_CASE( all1_convolve_newapi )
// {
//   using namespace cuda_convolve;

  

//   float sum_original = std::accumulate(image_folded_by_all1_.data(), image_folded_by_all1_.data() + image_.num_elements(),0.f);
//   inplace_gpu_convolution(image_.data(), &image_dims_[0], 
//   			  all1_kernel_.data(),&kernel_dims_[0],
//   			  selectDeviceWithHighestComputeCapability());

  

//   float sum = std::accumulate(image_.data(), image_.data() + image_.num_elements(),0.f);
//   // BOOST_CHECK_CLOSE(sum, sum_original, .00001);
// try{
//     BOOST_REQUIRE_CLOSE(sum, sum_original, .00001);
//   }
//   catch(...){
//     std::cout << boost::unit_test::framework::current_test_case().p_name << "\n"
// 	      << "expected:\n" << image_folded_by_all1_ << "\n"
// 	      << "received:\n" << image_ << "\n";
//   }


// }
//BOOST_AUTO_TEST_SUITE_END()
