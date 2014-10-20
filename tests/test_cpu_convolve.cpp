#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE CPU_CONVOLUTION
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>
#include <vector>
#include <functional>
#include "cpu_convolve.hpp"

using namespace anyfold;

BOOST_FIXTURE_TEST_SUITE( simple_cpu_convolution, anyfold::default_3D_fixture )

BOOST_AUTO_TEST_CASE( trivial_convolve )
{


  std::vector<float> kernel(kernel_size_);
  std::fill(kernel.begin(), kernel.end(), 0.f);

  anyfold::discrete_convolve_3d(image_.data(), &image_dims_[0],
				kernel.begin(), &kernel_dims_[0],
				one_.data());

  unsigned long sum = std::accumulate(one_.data(),one_.data() + image_size_,0.f);
  unsigned long checksum = std::accumulate(image_.data(),image_.data() + image_size_,0.f);
  
  BOOST_CHECK_LT(sum, checksum);

  unsigned center_pixel = image_size_/2 + 
    std::accumulate(image_dims_.begin(), image_dims_.end()-1,1.f,std::multiplies<int>())/2 +
    image_dims_[0]/2
    ;

  BOOST_CHECK_EQUAL(one_.data()[center_pixel], 0);
}

BOOST_AUTO_TEST_CASE( identity_convolve )
{

  anyfold::discrete_convolve_3d(image_.data(), &image_dims_[0],
				identity_kernel_.data(), &kernel_dims_[0],
				one_.data());

  unsigned long sum = std::accumulate(one_.data(),one_.data() + image_size_,0.f);
  unsigned long checksum = std::accumulate(image_.data(),image_.data() + image_size_,0.f);
  
  BOOST_CHECK_EQUAL(sum, checksum);

  unsigned center_pixel = image_size_/2 + 
    std::accumulate(image_dims_.begin(), image_dims_.end()-1,1.f,std::multiplies<int>())/2 +
    image_dims_[0]/2
    ;

  try{
    BOOST_REQUIRE_EQUAL(one_.data()[center_pixel ], image_.data()[center_pixel]);
  }
  catch(...){
    std::cerr << "input :\n" <<  image_ << "\n\n"
	      << "kernel:\n" <<  identity_kernel_ << "\n\n"
	      << "output:\n" <<  one_ << "\n";
  }

  BOOST_CHECK_EQUAL_COLLECTIONS(one_.data(),one_.data() + image_size_,
			       image_.data(),image_.data() + image_size_);
}

BOOST_AUTO_TEST_SUITE_END()


BOOST_FIXTURE_TEST_SUITE( mono_directional_convolutions, anyfold::default_3D_fixture )


BOOST_AUTO_TEST_CASE( horizontal_convolve )
{

  std::vector<int> padded_dims(3);
  std::copy(padded_image_.shape(), padded_image_.shape() + 3,padded_dims.begin());

  anyfold::discrete_convolve_3d(padded_image_.data(), &padded_dims[0],
				horizontal_kernel_.data(), &kernel_dims_[0],
				padded_one_.data());

  one_ = padded_one_[boost::indices[range(kernel_dims_[0]/2, padded_dims[0] - kernel_dims_[0]/2)][range(kernel_dims_[1]/2, padded_dims[1] - kernel_dims_[1]/2)][range(kernel_dims_[2]/2, padded_dims[1] - kernel_dims_[2]/2)] ];

  try{
  BOOST_REQUIRE_EQUAL_COLLECTIONS(one_.data(),one_.data() + image_size_,
				  image_folded_by_horizontal_ .data(),image_folded_by_horizontal_ .data() + image_size_);
    }
  catch(...){
    std::cerr << "input :\n" <<  image_ << "\n\n"
	      << "kernel:\n" <<  horizontal_kernel_ << "\n\n"
	      << "output:\n" <<  one_ << "\n"
      	      << "expected:\n" <<  image_folded_by_horizontal_ << "\n"
      ;
  }

}

BOOST_AUTO_TEST_CASE( vertical_convolve )
{

  std::vector<int> padded_dims(3);
  std::copy(padded_image_.shape(), padded_image_.shape() + 3,padded_dims.begin());

  anyfold::discrete_convolve_3d(padded_image_.data(), &padded_dims[0],
				vertical_kernel_.data(), &kernel_dims_[0],
				padded_one_.data());

  one_ = padded_one_[boost::indices[range(kernel_dims_[0]/2, padded_dims[0] - kernel_dims_[0]/2)][range(kernel_dims_[1]/2, padded_dims[1] - kernel_dims_[1]/2)][range(kernel_dims_[2]/2, padded_dims[1] - kernel_dims_[2]/2)] ];

  try{
  BOOST_REQUIRE_EQUAL_COLLECTIONS(one_.data(),one_.data() + image_size_,
				  image_folded_by_vertical_ .data(),image_folded_by_vertical_ .data() + image_size_);
    }
  catch(...){
    std::cerr << "input :\n" <<  image_ << "\n\n"
	      << "kernel:\n" <<  vertical_kernel_ << "\n\n"
	      << "output:\n" <<  one_ << "\n"
      	      << "expected:\n" <<  image_folded_by_vertical_ << "\n"
      ;
  }

}


BOOST_AUTO_TEST_CASE( depth_convolve )
{

  std::vector<int> padded_dims(3);
  std::copy(padded_image_.shape(), padded_image_.shape() + 3,padded_dims.begin());

  anyfold::discrete_convolve_3d(padded_image_.data(), &padded_dims[0],
				depth_kernel_.data(), &kernel_dims_[0],
				padded_one_.data());

  one_ = padded_one_[boost::indices[range(kernel_dims_[0]/2, padded_dims[0] - kernel_dims_[0]/2)][range(kernel_dims_[1]/2, padded_dims[1] - kernel_dims_[1]/2)][range(kernel_dims_[2]/2, padded_dims[1] - kernel_dims_[2]/2)] ];

  try{
  BOOST_REQUIRE_EQUAL_COLLECTIONS(one_.data(),one_.data() + image_size_,
				  image_folded_by_depth_ .data(),image_folded_by_depth_ .data() + image_size_);
    }
  catch(...){
    std::cerr << "input :\n" <<  image_ << "\n\n"
	      << "kernel:\n" <<  depth_kernel_ << "\n\n"
	      << "output:\n" <<  one_ << "\n"
      	      << "expected:\n" <<  image_folded_by_depth_ << "\n"
      ;
  }

}


BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE( complex_cpu_convolution, anyfold::default_3D_fixture )

BOOST_AUTO_TEST_CASE( asymm_cross_convolve )
{


  std::vector<int> padded_dims(3);
  std::copy(asymm_padded_image_.shape(), asymm_padded_image_.shape() + 3,padded_dims.begin());

  anyfold::discrete_convolve_3d(asymm_padded_one_.data(), &padded_dims[0],
				asymm_cross_kernel_.data(), &asymm_kernel_dims_[0],
				asymm_padded_image_.data());

  image_ = asymm_padded_image_[boost::indices[range(asymm_kernel_dims_[0]/2, padded_dims[0] - asymm_kernel_dims_[0]/2)][range(asymm_kernel_dims_[1]/2, padded_dims[1] - asymm_kernel_dims_[1]/2)][range(asymm_kernel_dims_[2]/2, padded_dims[2] - asymm_kernel_dims_[2]/2)] ];

  try{
    BOOST_REQUIRE_EQUAL_COLLECTIONS(image_.data(),image_.data() + image_size_,
				    one_folded_by_asymm_cross_kernel_ .data(), one_folded_by_asymm_cross_kernel_.data() + image_size_);
  }
  catch(...){
    std::cerr << "input :\n" <<  one_ << "\n\n"
	      << "kernel:\n" <<  asymm_cross_kernel_ << "\n\n"
	      << "output:\n" <<  image_ << "\n"
      	      << "expected:\n" <<  one_folded_by_asymm_cross_kernel_ << "\n"
	      << "padded_output:\n" << asymm_padded_image_ << "\n"
      ;
  }

}

BOOST_AUTO_TEST_CASE( asymm_identity_convolve )
{


  std::vector<int> padded_dims(3);
  std::copy(asymm_padded_image_.shape(), asymm_padded_image_.shape() + 3,padded_dims.begin());
  
  anyfold::discrete_convolve_3d(asymm_padded_one_.data(), &padded_dims[0],
				asymm_identity_kernel_.data(), &asymm_kernel_dims_[0],
				asymm_padded_image_.data());

  image_ = asymm_padded_image_[boost::indices[range(asymm_kernel_dims_[0]/2, padded_dims[0] - asymm_kernel_dims_[0]/2)][range(asymm_kernel_dims_[1]/2, padded_dims[1] - asymm_kernel_dims_[1]/2)][range(asymm_kernel_dims_[2]/2, padded_dims[2] - asymm_kernel_dims_[2]/2)] ];

  try{
    BOOST_REQUIRE_EQUAL_COLLECTIONS(image_.data(),image_.data() + image_size_,
				    one_folded_by_asymm_identity_kernel_ .data(), one_folded_by_asymm_identity_kernel_.data() + image_size_);
  }
  catch(...){
    std::cerr << "input :\n" <<  one_ << "\n\n"
	      << "kernel:\n" <<  asymm_identity_kernel_ << "\n\n"
	      << "output:\n" <<  image_ << "\n"
      	      << "expected:\n" <<  one_folded_by_asymm_identity_kernel_ << "\n"
	      << "padded_output:\n" << asymm_padded_image_ << "\n"
      ;
  }

}

BOOST_AUTO_TEST_CASE( asymm_one_convolve )
{


  std::vector<int> padded_dims(3);
  std::copy(asymm_padded_image_.shape(), asymm_padded_image_.shape() + 3,padded_dims.begin());

  anyfold::discrete_convolve_3d(asymm_padded_one_.data(), &padded_dims[0],
				asymm_one_kernel_.data(), &asymm_kernel_dims_[0],
				asymm_padded_image_.data());

  image_ = asymm_padded_image_[boost::indices[range(asymm_kernel_dims_[0]/2, padded_dims[0] - asymm_kernel_dims_[0]/2)][range(asymm_kernel_dims_[1]/2, padded_dims[1] - asymm_kernel_dims_[1]/2)][range(asymm_kernel_dims_[2]/2, padded_dims[2] - asymm_kernel_dims_[2]/2)] ];

  try{
    BOOST_REQUIRE_EQUAL_COLLECTIONS(image_.data(),image_.data() + image_size_,
				     one_folded_by_asymm_one_kernel_ .data(), one_folded_by_asymm_one_kernel_.data() + image_size_);
  }
  catch(...){
    std::cerr << "input :\n" <<  one_ << "\n\n"
	      << "kernel:\n" <<  asymm_one_kernel_ << "\n\n"
	      << "output:\n" <<  image_ << "\n"
      	      << "expected:\n" <<  one_folded_by_asymm_one_kernel_ << "\n"
	      << "padded_output:\n" << asymm_padded_image_ << "\n"
      ;
  }

}
BOOST_AUTO_TEST_SUITE_END()
