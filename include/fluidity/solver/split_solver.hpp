//==--- fluidity/solver/simulation_updater.hpp ---------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  simulation_updater.hpp
/// \brief This file defines a class which updates a simulation, and the
///        implementation is specialized for CPU and GPU execution policies.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_SPLIT_SOLVER_HPP
#define FLUIDITY_SOLVER_SPLIT_SOLVER_HPP

namespace fluid  {
namespace solver {

namespace detail {

enum class Input { left = 0, right = 1 };
template <Input Direction> struct InputSelector {};

/// Alias for creating a left input state.
using left_input_t = InputSelector<Input::left>;
/// Aliad for creating a right input state.
using right_input_t = InputSelector<Input::right>;
}

/// The Split solver class defines a functor which updates states using a split
/// method. It can be specialized for different systems of different dimension.
/// This implementation is defaulted for the 1D case.
/// \tparam Dimensions The number of dimensions to solve over.
template <typename Traits, std::size_t Dimensions = 1>
struct SplitSolver {
 private:
  /// Defines the traits of the solver.
  using traits_t        = std::decay_t<Traits>;
  /// Defines the type of the loader for the data.
  using loader_t        = typename traits_t::loader_t;
  /// Defines the type of the reconstructor of the data.
  using reconstructor_t = typename traits_t::reconstructor_t;
  /// Defines the type of the solver for the fluxes between cells.
  using flux_solver_t   = typename traits_t::flux_solver_t;

  /// Alias for creating a left input state.
  using left_input_t  = detail::InputSelector<detail::Input::left>;
  /// Aliad for creating a right input state.
  using right_input_t = detail::InputSelector<detail::Input::right>;

  /// Defines the number of dimensions to solve over.
  static constexpr std::size_t num_dimensions = 1;
  /// Defines the amount of padding in the data loader.
  static constexpr std::size_t padding        = Loader::padding;

 public:
  /// Solve function which invokes the solver.
  /// \param[in] data     The iterator which points to the start of the global
  ///            state data. If the iterator does not have 1 dimension then a
  ///            compile time error is generated.
  /// \param[in] flux     An iterator which points to the flux to update. 
  /// \tparam    Iterator The type of the iterator.
  /// \tparam    Flux     The type of the flux iterator.
  template <typename Iterator, typename Flux>
  fluidity_device_only void solve(Iterator&& data, Flux&& flux) const
  {
    static_assert(std::decay_t<Iterator>::num_dimensions() == num_dimensions,
                  "Dimensions of iterator do not match solver specialization");
    auto loader      = loader_t{};
    auto global_iter = get_global_iterator(data);
    auto patch_iter  = get_patch_iterator(data);
    auto size        = begin.size(dim_x);

    // Load the global data into the shared data:
    *patch_iter = *global_iter;

    // Load in the padding and boundary data:
    loader.load_internal(patch_iter, dim);
    loader.load_boundary(global_iter, patch_iter, dim);
    __syncthreads();

    // Run the rest of the sovler .. =D
    auto in_fwrd =
      reconstructor_t{}(patch_iter, material_t{}, params.dtdh, dim_x);
    auto in_back =
      reconstructor_t{}(
        patch_iter.offset(-1, dim_x), material_t{}, params.dtdh, dim_x);

    auto flux_diff = 
      flux_solver_t{}(in_fwrd.left, in_fwrd.right, material_t{}, dim_x) -
      flux_solver_t{}(in_back.left, in_back.right, material_t{}, dim_x);
  }

 private:
  /// Returns a reconstructed left input for the flux solver.
  fluidity_device_only make_recon_input(Iterator&& it, left_input_t)
  {
    return reconstructor_t{}()
  }

  /// Returns a global multi dimensional iterator which is shifted to the global
  /// thread index in the x-dimension.
  /// \param[in] it       The iterator to the start of the global data.
  /// \tparam    Iterator The type of the iterator.
  template <typename Iterator>
  fluidity_device_only get_global_iterator(Iterator&& it) const
  {
    using state_t    = std::decay_t<decltype(*it)>;
    using dim_info_t = DimInfo<num_dimensions>;

    auto it = make_multidim_iterator<state_t>(&(*it), dim_info_t{size});
    return it.offset(flattened_id(dim_x), dim_x);
  }

  /// Returns a shared multi dimensional iterator which is offset by the amount
  /// of padding and the local thread index, so the iterator points to the
  /// appropriate data for the thread to operate on.
  /// \param[in] it       The iterator to the start of the global data.
  /// \tparam    Iterator The type of the iterator.
  template <typename Iterator>
  fluidity_device_only get_patch_iterator(Iterator&& it) const
  {
    using state_t    = std::decay_t<decltype(*it)>;
    using dim_info_t = DimInfoCt<default_threads_per_block>;

    auto it = make_multidim_iterator<state_t, dim_info_t>();
    return it.offset(padding + thread_id(dim_x), dim_x);
  }
};

}} // namespace fluid::solver


#endif // FLUIDITY_SOLVER_SPLIT_SOLVER_HPP