#include <fluidity/dimension/thread_index.hpp>

#if defined(__CUDACC__)

namespace fluid  {
namespace detail {

/*

/// Implementation for the x dimension case for the thread index.
fluidity_device_only inline std::size_t thread_id_impl(dim_x)
{
  return threadIdx.x;
}

/// Implementation for the x dimension case for the thread index.
fluidity_device_only std::size_t thread_id_impl(dim_y)
{
  return threadIdx.y;
}

/// Implementation for the x dimension case for the thread index.
fluidity_device_only std::size_t thread_id_impl(dim_z)
{
  return threadIdx.z;
}


/// Implementation for the x dimension case for the flattened index.
fluidity_device_only std::size_t flattened_id_impl(dim_x)
{
  return threadIdx.x + blockIdx.x * blockDim.x;
}

/// Implementation for the y dimension case for the flattened index.
fluidity_device_only std::size_t flattened_id_impl(dim_y)
{
  return threadIdx.y + blockIdx.y * blockDim.y;
}

/// Implementation for the z dimension case for the flattened index.
fluidity_device_only std::size_t flattened_id_impl(dim_z)
{
  return threadIdx.z + blockIdx.z * blockDim.z;
}
*/
}} // namespace fluid::detail

#endif // __CUDACC__