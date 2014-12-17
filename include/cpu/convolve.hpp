#ifndef _CPU_CONVOLVE_HPP_
#define _CPU_CONVOLVE_HPP_
#include <sstream>
#include <numeric>
#include <functional>
#include "image_stack_utils.h"

namespace anyfold {

  namespace cpu {
    
    template <typename ImageStackT,typename CImageStackT, typename DimT>
    void convolve(CImageStackT& _image, 
		  CImageStackT& _kernel, 
		  ImageStackT& _result,
		  const std::vector<DimT>& _offset){


      if(!_image.num_elements())
	return;

      std::vector<DimT> half_kernel(3);
      for(unsigned i = 0;i<3;++i)
	half_kernel[i] = _kernel.shape()[i]/2;

      float image_value = 0;    
      float kernel_value = 0;    
      float value = 0;    

      for(int image_z = _offset[2];image_z<int(_image.shape()[2]-_offset[2]);++image_z){
	for(int image_y = _offset[1];image_y<int(_image.shape()[1]-_offset[1]);++image_y){
	  for(int image_x = _offset[0];image_x<int(_image.shape()[0]-_offset[0]);++image_x){

	    _result[image_x][image_y][image_z] = 0.f;
	  
	    image_value = 0;    
	    kernel_value = 0;   
	    value = 0;          
	  
	    for(int kernel_z = 0;kernel_z<int(_kernel.shape()[2]);++kernel_z){
	      for(int kernel_y = 0;kernel_y<int(_kernel.shape()[1]);++kernel_y){
		for(int kernel_x = 0;kernel_x<int(_kernel.shape()[0]);++kernel_x){

		  kernel_value  =  _kernel[_kernel.shape()[0]-1-kernel_x][_kernel.shape()[1]-1-kernel_y][_kernel.shape()[2]-1-kernel_z]	;
		  image_value   =  _image[image_x-half_kernel[0]+kernel_x][image_y-half_kernel[1]+kernel_y][image_z-half_kernel[2]+kernel_z]		;

		  value += kernel_value*image_value;
		}
	      }
	    }
	    _result[image_x][image_y][image_z] = value;
	  }
	}
      }


    }
  
    template <typename ExtentT, typename SrcIterT, typename KernIterT, typename OutIterT>
    void convolve_3d(SrcIterT src_begin, ExtentT* src_extents,
		     KernIterT kernel_begin, ExtentT* kernel_extents,
		     OutIterT out_begin)
    {
      std::vector<ExtentT> image_shape(src_extents,src_extents+3);
      std::vector<ExtentT> kernel_shape(kernel_extents,kernel_extents+3);
      
      anyfold::image_stack_cref image(src_begin, image_shape);
      anyfold::image_stack_cref kernel(kernel_begin, kernel_shape);
      anyfold::image_stack_ref output(out_begin, image_shape);

      std::vector<ExtentT> offsets(3);
      for (unsigned i = 0; i < offsets.size(); ++i)
	offsets[i] = kernel_shape[i]/2;
      
      return convolve(image,kernel,output,offsets);
    }


    template <typename ExtentT, typename SrcIterT, typename KernIterT, typename OutIterT>
    void discrete_convolve_3d(SrcIterT src_begin, ExtentT* src_extents,
			      KernIterT kernel_begin, ExtentT* kernel_extents,
			      OutIterT out_begin)
    {

      for(int d = 0;d<3;++d){
	if(kernel_extents[d] % 2 == 0){
	  std::ostringstream msg;
	  msg << "[anyfold::discrete_convolve_3d]\teven kernel dimensions found " 
	      << kernel_extents[0] << "x" << kernel_extents[1] << "x" << kernel_extents[2] << " NOT SUPPORTED\n";
	  throw std::runtime_error(msg.str().c_str());
	}
      }

    
      const unsigned long src_size = std::accumulate(src_extents, src_extents + 3, 1, std::multiplies<ExtentT>());
      const unsigned long src_frame_size = std::accumulate(src_extents, src_extents + 2, 1, std::multiplies<ExtentT>());
      //const unsigned long kernel_size = std::accumulate(kernel_extents, kernel_extents + 3, 1, std::multiplies<ExtentT>());

      std::copy(src_begin, src_begin + src_size, out_begin);
      std::vector<unsigned long> src_indices;
      src_indices.reserve(src_size);

      //FIXME: not storage order independent, assuming row-major of all input containers
      //create index list input (strategy dependent: assume not padding for now)
      std::vector<long> half_kernel_dims(kernel_extents, kernel_extents + 3);
      std::transform(half_kernel_dims.begin(), half_kernel_dims.end(), half_kernel_dims.begin(),
		     std::bind2nd(std::divides<ExtentT>(), 2));

      unsigned long src_index;
    
    
      for(unsigned long z_index = half_kernel_dims[2];z_index<=(src_extents[2] - half_kernel_dims[2]);++z_index){
	for(unsigned long y_index = half_kernel_dims[1];y_index<=(src_extents[1]- half_kernel_dims[1]);++y_index){
	  for(unsigned long x_index = half_kernel_dims[0];x_index<=(src_extents[0]- half_kernel_dims[0]);++x_index){
	  
	    src_index = z_index*src_frame_size + y_index*src_extents[0] + x_index;
      
	    // is_bulk_index = (z_index >= half_kernel_dims[2] && z_index < (src_extents[2] - half_kernel_dims[2])) &&
	    //   (y_index >= half_kernel_dims[1] && y_index < (src_extents[1] - half_kernel_dims[1])) &&
	    //   (x_index >= half_kernel_dims[0] && x_index < (src_extents[0] - half_kernel_dims[0]));

	    // if(is_bulk_index){
	    src_indices.push_back(src_index);
	    // }
	  }
	}
      }
    
      //perform convolution
      long global_offset = 0;
      long kernel_offset = 0;
      long z,y,x,t;
    
      for(unsigned long bulk_index = 0;bulk_index<src_indices.size();++bulk_index){
	OutIterT value = out_begin + src_indices[bulk_index];
	*value = 0;
      
	// SrcIterT in = src_begin + src_indices[bulk_index];
	// KernIterT kernel_rev_iterator = kernel_begin + kernel_size - 1;      
	KernIterT kernel = kernel_begin;

	for(long z_offset = -1*half_kernel_dims[2];z_offset<=half_kernel_dims[2];++z_offset){
	  for(long y_offset = -1*half_kernel_dims[1];y_offset<=half_kernel_dims[1];++y_offset){
	    for(long x_offset = -1*half_kernel_dims[0];x_offset<=half_kernel_dims[0];++x_offset){

	      z = src_indices[bulk_index]/src_frame_size;
	      t = src_indices[bulk_index] - (z*src_frame_size);
	      y = t/src_extents[0];
	      x = t - (y*src_extents[0]);
	    
	      global_offset = (z + z_offset)*src_frame_size + (y+y_offset)*src_extents[0] + x+ x_offset;
	      // global_offset = z_offset*src_frame_size + y_offset*src_extents[0] + x_offset;
	      kernel_offset = (-1*z_offset + half_kernel_dims[2])*kernel_extents[1]*kernel_extents[0] + (-1*y_offset + half_kernel_dims[1])*kernel_extents[0] + (-1*x_offset + half_kernel_dims[0]); 
	      // *value += *(in + global_offset) * *(kernel_rev_iterator--);
	      *value += *(src_begin + global_offset) * *(kernel + kernel_offset);

	    }
	  }
	}
      }
    }

  };
};

#endif /* _CPU_CONVOLVE_H_ */
