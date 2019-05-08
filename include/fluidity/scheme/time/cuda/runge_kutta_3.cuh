//==--- fluidity/solver/time/cuda/runge_kutta_3.cuh -------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  runge_kutta_3.hpp
/// \brief This file provides a cuda implementation of the Runge-Kutta-3 method.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_TIME_CUDA_RUNGE_KUTTA_3_CUH
#define FLUIDITY_SOLVER_TIME_CUDA_RUNGE_KUTTA_3_CUH

#include <fluidity/execution/execution_policy.hpp>
#include <fluidity/iterator/multidim_iterator.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid  {
namespace solver {
namespace time   {
namespace cuda   {


template <typename Iterator, typename T, typename F, typename... Args>
fludity_global void 
runge_kutta_3(Iterator in_it, Iterator out_it, T dt, F f, Args... args)
{
  using it_t   = std::decay_t<Iterator>;
  using data_t = std::decay_t<decltype(*in_it)>;
  constexpr auto width = f.width(); 

  auto shared_it_in  = make_multidim_iterator<width>(in_it);
  auto shared_it_out = make_multidim_iterator<width>(out_it);

  unrolled_for<it_t::dimensions>([&] (auto dim)
  {
    apply([&] (auto& it)
    {
      it.shift(flattened_id(dim) + width, dim);
    }, in_it, out_it, shared_it_in, shared_it_out);
  });

  // Load the shared data;
  auto phi_n     = *in_it;
  *shared_it_in  = phi_n;
  *shared_it_out = phi_n;

  // Compute first temp evolution to t^{n+1}. We have to sync here because the
  // stencil in the next evolution needs to access the data from other threads.
  *shared_it_out = *shared_it_in - dt * f(shared_it_in, args...);
  __syncthreads();

  // Evolve again in time to t^{n+2} and then use the result in a weighted
  // average to compute phi^{n+ 1/2}. Again we need to sync for the next
  // evolution.
  *shared_it_in = 0.75 * phi_n 
                + 0.25 * (*shared_it_out - dt(shared_it_out, args...));
  __syncthreads();

  // Compute evolution to t^{n + 3/2}:
  *shared_it_out = *shared_it_in + dt * f(shared_it_in, args...);

  // Finally, set the output data:
  *out_it = (phi_n / 3.0) + (2.0 / 3.0 * (*shared_it_out));
}

/// Cuda interface for a Runge-Kutta-3 implementation. This simply computes the
/// blocks sizes from the \p in_it and then launches the cuda kernel, forwarding
/// the arguments.
/// \param[in]  in_it   The input iterator over the data to update.
/// \param[out] out_it  The output iterator to write the data to.
/// \param[in]  dt      The time delta for the evolution.
/// \param[in]  f       The function to use to perform the update.
/// \param[in]  args    Additional arguments for the function.
/// \tparam     It      The type of the input and output iterators.
/// \tparam     T       The type of the timestep.
/// \tparam     F       The type of the function.
/// \tparam     Args    The types of the arguments,
template <typename It, typename T, typename F, typename... Args>
void runge_kutta_3(It&& in_it, It&& out_it, Tdt, F&& f, Args&&... args)
{
  auto threads = exec::get_thread_sizes(in_it);
  auto blocks  = exec::get_block_sizes(in_it, threads);

  runge_kutta_3_impl<<<blocks, threads>>>(in_it, out_it, dt, f, args...);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
}

}}}} // namespace fluid::solver::time::cuda

#endif // FLUIDITY_SOLVER_TIME_CUDA_RUNGE_KUTTA_3_CUH

