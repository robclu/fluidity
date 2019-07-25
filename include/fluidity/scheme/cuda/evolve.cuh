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
#include <fluidity/traits/device_traits.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid  {
namespace scheme {
namespace cuda   {
namespace detail {

/// Cuda implementation for evolving the \p in_it data using the \p evolver to
/// set the \p out_it data. After calling this method, the \p out_it data will
/// be evolved by \p dt from the \p in_it data, using the method implemented by
/// the \p evolver.
///
/// This overload is for the case that other iterable data (\p func_or_it) is
/// required for the evolution, and is a multi-dimensional iterator.
///
/// \param[in] evolver        The type of the evolver.    
/// \param[in] in_it          The input iterator.
/// \param[in] out_it         The output iterator.          
/// \param[in] dt             The time delta for the evolution.
/// \param[in] dh             The size of the spacial resolution.
/// \param[in] boundaries     The boundaries for the evolution.
/// \param[in] func_or_it     An functor/iterator to use for the evaluation.
/// \param[in] args           Additional argumens for the evolution.
/// \tparam    Evolver        The type of the evolver.
/// \tparam    InIterator     The type of the input iterator.
/// \tparam    OutIterator    The type of the output iterator.
/// \tparam    T              The data type for the deltas.
/// \tparam    BoundContainer The type of the boundary container.
/// \tparam    FuncOrIt       The type of the functor/additional iterator.
/// \tparam    Args           Additional argument types.
template <
  typename    Evolver       ,
  typename    InIterator    ,
  typename    OutIterator   ,
  typename    T             , 
  typename    BoundContainer,
  typename    FuncOrIt      ,
  typename... Args          ,
  multiit_enable_t<FuncOrIt> = 0
>
fluidity_global auto evolve(
  Evolver        evolver   ,
  InIterator     in_it     , 
  OutIterator    out_it    ,
  T              dt        ,
  T              dh        ,
  BoundContainer boundaries,
  FuncOrIt       func_or_it,
  Args...        args
) -> void {
  using in_it_t      = std::decay_t<InIterator>;
  using out_it_t     = std::decay_t<OutIterator>;
  using func_or_it_t = std::decay_t<FuncOrIt>;
  using data_t       = std::decay_t<decltype(*in_it)>;
  using evolver_t    = std::decay_t<Evolver>;

  // Check iterator compatibility:
  static_assert(
    in_it_t::dimensions == out_it_t::dimensions &&
    in_it_t::dimensions == func_or_it_t::dimensions,
    "Iterators must have the same dimensionality"
  );

  constexpr auto width = evolver_t().width();
  constexpr auto dims  = in_it_t::dimensions;

  // Create shared memory iterators and then offset iterators for each
  // dimension.
  auto shared_in_it  = make_multidim_iterator<in_it_t     , width>(in_it);
  auto shared_out_it = make_multidim_iterator<out_it_t    , width>(out_it);
  //auto shared_oth_it = make_multidim_iterator<func_or_it_t, width>(func_or_it);

  unrolled_for<dims>([&] (auto dim) {
    const auto global_offset = flattened_id(dim);
    const auto block_offset  = thread_id(dim) + width;

    in_it.shift(global_offset, dim);
    out_it.shift(global_offset, dim);
    func_or_it.shift(global_offset, dim);
    shared_in_it.shift(block_offset, dim);
    shared_out_it.shift(block_offset, dim);
    //shared_oth_it.shift(block_offset, dim);
  });
  *shared_in_it  = *in_it;
  *shared_out_it = *in_it;
  //*shared_oth_it = *func_or_it;

  // Load the padding data for the shared iterator, so that all cells have valid
  // data for the evolution computation. This uses the boundary types at the
  // start and end of the domain, and the global iterator for the rest.
  evolver.load_padding(
    in_it       ,
    shared_in_it,
    boundaries
  );
  evolver.load_padding(
    in_it        ,
    shared_out_it,
    boundaries
  );
  __syncthreads();

/*
  printf("Before: {%3lu} : {%4.4f, %4.4f, %4.4f}, {%4.4f, %4.4f, %4.4f} \n",
    flattened_id(0),
    *in_it, *out_it, *func_or_it,
    // *in_it, *out_it, *func_or_it
    *shared_in_it, *shared_out_it, *func_or_it
    // *shared_in_it, *shared_out_it, *shared_oth_it
  );
*/

  // Evolve the shared input to the shared output, and then copy the result back
  // to global memory.
  evolver.evolve(
    shared_in_it ,
    shared_out_it,
    //in_it,
    //out_it,
    dt           ,
    dh           ,
    //shared_oth_it,
    func_or_it,
    args...
  );

/*
  printf("After: {%3lu} : {%4.4f, %4.4f, %4.4f}, {%4.4f, %4.4f, %4.4f} \n",
    flattened_id(0),
    *in_it, *out_it, *func_or_it,
    // *in_it, *out_it, *func_or_it
    *shared_in_it, *shared_out_it, *func_or_it
    // *shared_in_it, *shared_out_it, *shared_oth_it
  );
*/
  *out_it = *shared_out_it;
}

} // namespace detail

/// Interface to invoke the cuda implementation for evolving the \p in_it data
/// using the \p evolver to set the \p out_it data. After calling this method,
/// the \p out_it data will be evolved by \p dt from the \p in_it data, using
/// the method implemented by the \p evolver.
///
/// This overload is for the case that additional data is required for the
/// evolution, in the form of either a multi-dimensional iterator or a functor,
/// in which case \p args are additional arguments for the functor.
///
/// \param[in] evolver        The type of the evolver.    
/// \param[in] in_it          The input iterator.
/// \param[in] out_it         The output iterator.          
/// \param[in] dt             The time delta for the evolution.
/// \param[in] dh             The size of the spacial resolution.
/// \param[in] boundaries     The boundaries for the evolution.
/// \param[in] func_or_it     A functor/iterator for the evolution.
/// \param[in] args           Additional argumens for the evolution.
/// \tparam    Evolver        The type of the evolver.
/// \tparam    InIterator     The type of the input iterator.
/// \tparam    OutIterator    The type of the output iterator.
/// \tparam    T              The data type for the deltas.
/// \tparam    BoundContainer The type of the boundary container.
/// \tparam    FuncOrIt       The type of the functor/iterator.
/// \tparam    Args           Additional argument types.
template <
  typename    Evolver       ,
  typename    InIterator    ,
  typename    OutIterator   ,
  typename    T             , 
  typename    BoundContainer,
  typename    FuncOrIt      ,
  typename... Args
>
auto evolve(
  Evolver&&        evolver   ,
  InIterator&&     in_it     , 
  OutIterator&&    out_it    ,
  T                dt        ,
  T                dh        ,
  BoundContainer&& boundaries,
  FuncOrIt&&       func_or_it,
  Args&&...        args
) -> void {
  auto threads = exec::get_thread_sizes(in_it);
  auto blocks  = exec::get_block_sizes(in_it, threads);

  detail::evolve<<<blocks, threads>>>(
    evolver   ,
    in_it     ,
    out_it    ,
    dt        ,
    dh        ,
    boundaries,
    func_or_it,
    args...
  );
  fluidity_check_cuda_result(cudaDeviceSynchronize());
}



}}} // namespace fluid::scheme::cuda


#endif // FLUIDITY_SCHEME_CUDA_EVOLVE_HPP
