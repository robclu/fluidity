//==--- fluidity/solver/split_solver.hpp ------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  split_solver.hpp
/// \brief This file defines implementations of a split solver.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_SPLIT_SOLVER_HPP
#define FLUIDITY_SOLVER_SPLIT_SOLVER_HPP

#include "solver_utilities.hpp"
#include <fluidity/dimension/dimension.hpp>
#include <fluidity/iterator/multidim_iterator.hpp>
#include <fluidity/utility/cuda.hpp>

namespace fluid  {
namespace solver {

/// The SplitSolver class defines an implementation of a solver which updates
/// states using a dimensionally split method. It can be specialized for
/// systems of different dimension. This implementation is defaulted for the 1D
/// case.
/// \tparam Traits     The components used by the solver.
/// \tparam Dimensions The number of dimensions to solve over.
template <typename Traits, std::size_t Dimensions = 1>
struct SplitSolver {
 private:
  /// Defines the traits of the solver.
  using traits_t         = std::decay_t<Traits>;
  /// Defines the type of the loader for the data.
  using loader_t         = typename traits_t::loader_t;
  /// Defines the type of the reconstructor of the data.
  using reconstructor_t  = typename traits_t::reconstructor_t;
  /// Defines the type of the evaluator for the fluxes between cells.
  using flux_evaluator_t = typename traits_t::flux_evaluator_t;

  /// Defines an instance of the flux evaluator.
  static constexpr auto flux_evaluator = flux_evaluator_t{};

  /// Alias for creating a left input state.
  static constexpr auto back_input = back_input_t{};
  /// Aliad for creating a right input state.
  static constexpr auto fwrd_input = fwrd_input_t{};

  /// Defines the number of dimensions to solve over.
  static constexpr std::size_t num_dimensions = 1;
  /// Defines the amount of padding in the data loader.
  static constexpr std::size_t padding        = loader_t::padding;

 public:
  /// Solve function which invokes the solver.
  /// \param[in] data     The iterator which points to the start of the global
  ///            state data. If the iterator does not have 1 dimension then a
  ///            compile time error is generated.
  /// \param[in] flux     An iterator which points to the flux to update. 
  /// \tparam    Iterator The type of the iterator.
  /// \tparam    Flux     The type of the flux iterator.
  template <typename It, typename M, typename T>
  fluidity_device_only void solve(It&&                  in      ,
                                  It&&                  out     ,
                                  M                     material,
                                  T                     dtdh    ,
                                  const BoundarySetter& setter  ) const
  {
    //const auto loader = loader_t{};
    auto global_iter  = get_global_iterator(in);
    auto patch_iter   = get_patch_iterator(in);

    // Load the global data into the shared data:
    *patch_iter = *global_iter;

    // Load in the padding and boundary data:
    //loader.load_patch(patch_iter, dim_x);
    loader_t::load_boundary(global_iter, patch_iter, dim_x, setter);
    __syncthreads();

    if (flattened_id(dim_x) == 0)
    {
      auto s  = *global_iter;
      auto s1 = *patch_iter;
      printf("\n| d: %3.3f | p: %3.3f | v_x : %3.3f |", s[0], s[1], s[2]);
      printf("\n| d: %3.3f | p: %3.3f | v_x : %3.3f |\n-=-=-=\n", s1[0], s1[1], s1[2]);
    }

    // Run the rest of the sovler .. =D
    auto in_fwrd = make_recon_input(patch_iter, material, dtdh, fwrd_input);
    auto in_back = make_recon_input(patch_iter, material, dtdh, back_input);

    if (flattened_id(dim_x) == 0)
    {
      auto s  = *global_iter;
      auto s1 = *patch_iter;
      printf("| d: %3.3f | p: %3.3f | v_x : %3.3f |", s[0], s[1], s[2]);
      printf("\n| d: %3.3f | p: %3.3f | v_x : %3.3f |\n-=-=-=\n", s1[0], s1[1], s1[2]);
    }

    global_iter = get_global_iterator(out);


    auto f = flux_evaluator(in_fwrd.left, in_fwrd.right, material, dim_x);
    auto b = flux_evaluator(in_fwrd.left, in_fwrd.right, material, dim_x);
    if (flattened_id(dim_x) == 0)
    {
      printf("| d: %3.3f | p: %3.3f | v_x : %3.3f |", f[0], f[1], f[2]);
      printf("\n| d: %3.3f | p: %3.3f | v_x : %3.3f |\n-=-=-=\n", b[0], b[1], b[2]);
    }

    *global_iter = *patch_iter - dtdh * 
      (flux_evaluator(in_fwrd.left, in_fwrd.right, material, dim_x) -
       flux_evaluator(in_back.left, in_back.right, material, dim_x));


    if (flattened_id(dim_x) == 0)
    {
      auto s = *global_iter;
      printf("\n| d: %3.3f | p: %3.3f | v_x : %3.3f |\n", s[0], s[1], s[2]);
      printf("End\n--------------\n");
    }
  }

 private:
  /// Returns a reconstructed left input for the flux solver.
  template <typename It, typename M, typename T>
  fluidity_device_only decltype(auto)
  make_recon_input(It&& it, M mat, T dtdh, back_input_t) const
  {
    return reconstructor_t{}(it.offset(-1, dim_x), mat, dtdh, dim_x);
  }

  /// Returns a reconstructed left input for the flux solver.
  template <typename It, typename M, typename T>
  fluidity_device_only decltype(auto)
  make_recon_input(It&& it, M mat, T dtdh, fwrd_input_t) const
  {
    return reconstructor_t{}(it, mat, dtdh, dim_x);
  }

  /// Returns a global multi dimensional iterator which is shifted to the global
  /// thread index in the x-dimension.
  /// \param[in] it       The iterator to the start of the global data.
  /// \tparam    Iterator The type of the iterator.
  template <typename Iterator>
  fluidity_device_only auto get_global_iterator(Iterator&& it) const
  {
    using dim_info_t = DimInfo<num_dimensions>;
    auto output_it   = make_multidim_iterator(&(*it)                    ,
                                              dim_info_t{it.size(dim_x)});
    return output_it.offset(flattened_id(dim_x), dim_x);
  }

  /// Returns a shared multi dimensional iterator which is offset by the amount
  /// of padding and the local thread index, so the iterator points to the
  /// appropriate data for the thread to operate on.
  /// \param[in] it       The iterator to the start of the global data.
  /// \tparam    Iterator The type of the iterator.
  template <typename Iterator>
  fluidity_device_only auto get_patch_iterator(Iterator&& it) const
  {
    using state_t    = std::decay_t<decltype(*(it))>;
    using dim_info_t = DimInfoCt<default_threads_per_block>;

    auto output_it = make_multidim_iterator<state_t, dim_info_t, padding>();
    return output_it.offset(thread_id(dim_x) + padding, dim_x);
  }
};

}} // namespace fluid::solver


#endif // FLUIDITY_SOLVER_SPLIT_SOLVER_HPP