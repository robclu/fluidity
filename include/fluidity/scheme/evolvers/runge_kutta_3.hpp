//==--- fluidity/solver/time/runge_kutta_3.hpp ------------- -*- C++ -*- ---==//
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
/// \brief This file defines the interface for a Runge-Kutta-3 scheme.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_TIME_RUNGE_KUTTA_3_HPP
#define FLUIDITY_SOLVER_TIME_RUNGE_KUTTA_3_HPP

#include "cuda/runge_kutta_3.cuh"

namespace fluid  {
namespace solver {
namespace time   {

/// Interface for time evolution of the \p in_it iterable data using the RK3
/// method. The \p f function operates on the iterator data such that the
/// following evolution is performed:
///
///   $ \phi^{n+1} = \phi^n - \Delta t H(\phi^n)$
///
/// Where \p in_it is the data $\phi^n$ data, \p out_it is the output data
/// $\phi^{n+1}$, and $H$ is the predicate function. The RK3 implementation here
/// takes the following steps:
///
/// - Take two temporate steps to compute $\widetilde{\phi}^{n+2}$ as:
///     $\widetilde{\phi}^{n+1} = \phi^n - \Delta t H(\phi^n)$
///     $\widetilde{\phi}^{n+2} = \widetilde{\phi}^{n+1} - 
///                             - \Delta t H(\widetilde{\phi}^{n+1})$
///
/// - Take a weighted average as:
///     $\widetilde{\phi}^{n+\frac{1}{2}} = \frac{3}{4} \phi^n
///                                       + \frac{1}{4} \widetilde{\phi}^{n+2}$
///
/// - Take a temporary step to compute $\widetilde{\phi}^{n + \frac{3}{2}}$ as:
///     $\widetilde{\phi}^{n+\frac{3}{2}} 
///         = \widetilde{\phi}^{n + \frac{1}{2}}
///         - \Delta t H(\widetilde{\phi})^{n + \frac{1}{2}}$
///
/// - Finally, perform the final update as:
///     $\phi^{n+1} = \frac{1}{3} \phi^n
///                 + \frac{2}{3} \widetilde{\phi}^{n + \frac{3}{2}}$
///
/// This overload is enabled when the iterators have a cpu execution policy.
///
/// \param[in]  in_it   The input iterator over the data to update.
/// \param[out] out_it  The output iterator to write the data to.
/// \param[in]  dt      The time delta for the evolution.
/// \param[in]  f       The function to use to perform the update.
/// \param[in]  args    Additional arguments for the function.
/// \tparam     It      The type of the input and output iterators.
/// \tparam     T       The type of the timestep.
/// \tparam     F       The type of the function.
/// \tparam     Args    The types of the arguments.
template <typename     It  ,
          typename     T   ,
          typename     F   ,
          typename...  Args, exec::cpu_enable_t<Iterator> = 0>
void runge_kutta_3(It&& in_it, It&& out_it, T dt, F&& f, Args&&... args)
{
  static_assert(false         , "CPU implementation is not yet available!");
  static_assert(is_scheme_v<F>, "Function must conform to Scheme interface!");
}


/// Interface for time evolution of the \p in_it iterable data using the RK3
/// method. The \p f function operates on the iterator data such that the
/// following evolution is performed:
///
///   $ \phi^{n+1} = \phi^n - \Delta t H(\phi^n)$
///
/// Where \p in_it is the data $\phi^n$ data, \p out_it is the output data
/// $\phi^{n+1}$, and $H$ is the predicate function. The RK3 implementation here
/// takes the following steps:
///
/// - Take two temporate steps to compute $\widetilde{\phi}^{n+2}$ as:
///     $\widetilde{\phi}^{n+1} = \phi^n - \Delta t H(\phi^n)$
///     $\widetilde{\phi}^{n+2} = \widetilde{\phi}^{n+1} - 
///                             - \Delta t H(\widetilde{\phi}^{n+1})$
///
/// - Take a weighted average as:
///     $\widetilde{\phi}^{n+\frac{1}{2}} = \frac{3}{4} \phi^n
///                                       + \frac{1}{4} \widetilde{\phi}^{n+2}$
///
/// - Take a temporary step to compute $\widetilde{\phi}^{n + \frac{3}{2}}$ as:
///     $\widetilde{\phi}^{n+\frac{3}{2}} 
///         = \widetilde{\phi}^{n + \frac{1}{2}}
///         - \Delta t H(\widetilde{\phi})^{n + \frac{1}{2}}$
///
/// - Finally, perform the final update as:
///     $\phi^{n+1} = \frac{1}{3} \phi^n
///                 + \frac{2}{3} \widetilde{\phi}^{n + \frac{3}{2}}$
///
/// This overload is enabled when the iterators have a gpu execution policy.
///
/// \param[in]  in_it   The input iterator over the data to update.
/// \param[out] out_it  The output iterator to write the data to.
/// \param[in]  dt      The time delta for the evolution.
/// \param[in]  f       The function to use to perform the update.
/// \param[in]  args    Additional arguments for the function.
/// \tparam     It      The type of the input and output iterators.
/// \tparam     T       The type of the timestep.
/// \tparam     F       The type of the function.
/// \tparam     Args    The types of the arguments.
template <typename     It  ,
          typename     T   ,
          typename     F   ,
          typename...  Args, exec::cpu_enable_t<It> = 0>
void runge_kutta_3(It&& in_it, It&& out_it, T dt, F&& f, Args&&... args)
{
  static_assert(is_scheme_v<F>, "Function must conform to scheme interface!");
  cuda::runge_kutta_3(std::forward<It>(in_it)    ,
                      std::forward<It>(out_it)   ,
                      dt                         ,
                      std::forward<F>(f)         ,
                      std::forward<Args>(args)...);
}
  
}}} // namespace fluid::solver::time

#endif // FLUIDITY_SOLVER_TIME_RUNGE_KUTTA_3_HPP
