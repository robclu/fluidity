//==--- fluidity/scheme/cuda/evolve.cuh -------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  evolve.cuh
/// \brief This file defines cuda implementation for evolving data.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_CUDA_EVOLVE_HPP
#define FLUIDITY_SCHEME_CUDA_EVOLVE_HPP

#include <fluidity/algorithm/apply.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid  {
namespace scheme {
namespace cuda   {

/// Interface for evolving the \p it_in data using the \p evolver to set the
/// \p out_it data.
/// \param[in] evl
/// \param[in] in
/// \param[in] out
/// \param[in] dt
/// \param[in] dh
/// \param[in] args
/// \tparam    Evolver
/// \tparam    ItIn
/// \tparam    ItOut
/// \tparam    T
/// \tparam    Args
template <typename    Evl  ,
          typename    ItIn ,
          typename    ItOut,
          typename    T    , 
          typename... Args , exec::gpu_enable_t<ItIn> = 0>
fluidity_global void
evolve_impl(Evl evl, ItIn in, ItOut out, T dt, T dh, Args... args)
{
  using it_in_t   = std::decay_t<ItIn>;
  using it_out_t  = std::decay_t<ItOut>;
  using data_t    = std::decay_t<decltype(*in)>;
  using evolver_t = std::decay_t<Evl>;

  // Check iterator compatibility:
  static_assert(it_in_t::dimensions == it_out_t::dimensions,
                "Iterators must have the same dimensionality");

  const     auto evolver = evolver_t();
  constexpr auto width   = evolver.width();
  constexpr auto dims    = it_in_t::dimensions;

  // Create shared memory iterators and then offset iterators for each
  // dimension.
  auto shared_it_in  = make_multidim_iterator<it_in_t ,width>(in);
  auto shared_it_out = make_multidim_iterator<it_out_t,width>(out); 

  unrolled_for<dims>([&] (auto dim)
  {
    in.shift(flattened_id(dim), dim);
    out.shift(flattened_id(dim), dim);
    shared_it_in.shift(flattened_id(dim) + width, dim);
    shared_it_out.shift(flattened_id(dim) + width, dim);
  });

  *shared_it_in  = *in;
  *shared_it_out = *in;
  __syncthreads();
 
  evolver.evolve(shared_it_in, shared_it_out, dt, dh, args...);

  // The evolver sets the shared_it_out data to the evolved value, which needs
  // to now be set in global memory.
  *out = *shared_it_out;
}

/// Interface for evolving the \p it_in data using the \p evolver to set the
/// \p out_it data.
/// \param[in] evl
/// \param[in] in
/// \param[in] out
/// \param[in] dt
/// \param[in] dh
/// \param[in] args
/// \tparam    Evolver
/// \tparam    ItIn
/// \tparam    ItOut
/// \tparam    T
/// \tparam    Args
template <typename    Evl  ,
          typename    ItIn ,
          typename    ItOut,
          typename    T    , 
          typename... Args , exec::gpu_enable_t<ItIn> = 0>
void evolve(Evl&& evl, ItIn&& in, ItOut&& out, T dt, T dh, Args&&... args)
{
  auto threads = exec::get_thread_sizes(in);
  auto blocks  = exec::get_block_sizes(in, threads);

  evolve_impl<<<blocks, threads>>>(evl, in, out, dt, dh, args...);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
}

}}} // namespace fluid::scheme::cuda


#endif // FLUIDITY_SCHEME_CUDA_EVOLVE_HPP
