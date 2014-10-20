#ifndef _CPU_CONVOLVE_HPP_
#define _CPU_CONVOLVE_HPP_
#include <sstream>
#include <numeric>
#include <functional>

namespace anyfold {

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
    const unsigned long kernel_size = std::accumulate(kernel_extents, kernel_extents + 3, 1, std::multiplies<ExtentT>());

    std::copy(src_begin, src_begin + src_size, out_begin);
    std::vector<unsigned long> src_indices;
    src_indices.reserve(src_size);

    //FIXME: not storage order independent, assuming row-major of all input containers
    //create index list input (strategy dependent: assume not padding for now)
    std::vector<long> half_kernel_dims(kernel_extents, kernel_extents + 3);
    std::transform(half_kernel_dims.begin(), half_kernel_dims.end(), half_kernel_dims.begin(),
		   std::bind2nd(std::divides<ExtentT>(), 2));

    unsigned long src_index;
    bool is_bulk_index = false;
    
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
      
      SrcIterT in = src_begin + src_indices[bulk_index];
      KernIterT kernel_rev_iterator = kernel_begin + kernel_size - 1;      
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

}

#endif /* _CPU_CONVOLVE_H_ */
