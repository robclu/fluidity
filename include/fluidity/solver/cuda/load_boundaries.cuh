//==--- fluidity/solver/load_boundaries.cuh ---------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  load_boundaries.cuh
/// \brief This file defines a cuda implementation kernel which loads the
///        boundary data for a global iterator.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_LOAD_BOUNDARIES_CUH
#define FLUIDITY_SOLVER_LOAD_BOUNDARIES_CUH

namespace fluid  {
namespace solver {
namespace detail {
namespace cuda   {

template <typename Loader, typename IT>
fluidity_global void load_boundaries_impl(IT&&                  it    ,
                                          const BoundarySetter& setter)
{
  using loader_t         = std::decay_t<Loader>;
  constexpr auto dims    = std::decay_t<IT>::dimensions;
  constexpr auto padding = loader_t::padding;

  unrolled_for<dims>([&] (auto dim)
  {
    it = it.offset(flattened_id(dim) + padding, dim);
  });
  loader_t::load_global_boundaries(it, setter);
}

template <typename Solver, typename IT>
void load_boundaries(Solver&& solver, IT&& it, const BoundarySetter& setter)
{
  using loader_t = std::decay_t<Solver>::loader_t;
  load_boundaries_impl<loader_t>
    <<<solver.block_sizes(), solver.thread_sizes()>>>(it, setter);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
}

}}}} // namespace fluid::solver::detail::cuda

#endif // FLUIDITY_SOLVER_LOAD_BOUNDARIES_CUH