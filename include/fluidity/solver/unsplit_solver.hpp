//==--- fluidity/solver/unsplit_solver.hpp ----------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  unsplit_solver.hpp
/// \brief This file defines implementations of an unsplit solver.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_UNSPLIT_SOLVER_HPP
#define FLUIDITY_SOLVER_UNSPLIT_SOLVER_HPP

#include "solver_utilities.hpp"

namespace fluid  {
namespace solver {

/// The UnsplitSolver class defines an implementatio of a solver which updates
/// states using a dimensionally unsplit method. It can be specialized for
/// systems of different dimension. This implementation is defaulted for the 1D
/// case.
/// \tparam Traits     The componenets used by the solver.
/// \tparam Dimensions The number of dimensions to solve over.
template <typename FluxSolver, typename Loader, typename Dims = Num<1>>
struct UnsplitSolver {
 private:
  /// Defines the type of the flux solver.
  using flux_solver_t = std::decay_t<FluxSolver>;
  /// Defines the type of the loader for the data.
  using loader_t      = std::decay_t<Loader>;
  /// Defines the type of the boundary setter.
  using setter_t      = BoundarySetter;
  /// Defines a reference type to the boundary setter.
  using setter_ref_t  = const BoundarySetter&;


  /// Defines the number of dimensions to solve over.
  static constexpr auto num_dimensions = std::size_t{Dims()};
  /// Defines the amount of padding in the data loader.
  static constexpr auto padding        = loader_t::padding;

 public:
  /// Creates the unsplit solver.
  /// \param[in] it The iterator over the computational space to solve.
  /// \tparam    It The type of the iterator.
  template <typename It>
  UnsplitSolver(It&& it) {}

  /// Solve function which invokes the solver.
  /// \param[in] data     The iterator which points to the start of the global
  ///            state data. If the iterator does not have 1 dimension then a
  ///            compile time error is generated.
  /// \param[in] flux     An iterator which points to the flux to update. 
  /// \tparam    Iterator The type of the iterator.
  /// \tparam    Flux     The type of the flux iterator.
  template <typename It, typename Mat, typename T, exec::gpu_enable_t<It> = 0>
  void solve(It&& in, It&& out, Mat&& mat, T dtdh, setter_ref_t setter) const 
  {

  }

  /// Overlaod of the call operator to invoke a pass of solving on the input and
  /// output data iterators for a specific dimension which is defined by Value.
  /// The data from the \p in iterator is used to compute the update which is
  /// then written to the \p out iterator.
  /// \param[in] in     The input multi dimensional iterator over state data.
  /// \param[in] out    The output multi dimensional iterator over state data.
  /// \param[in] mat    The material for the system.
  /// \tparam    It     The type of the iterator.
  /// \tparam    Mat    The type of the material for the system.
  /// \tparam    T      The data type for the scaling factor.
  /// \tparam    Value  The value which defines the dimension for the pass.
  template <typename It, typename Mat, typename T, typename Dim>
  fluidity_device_only static void invoke(It&&             in    ,
                                          It&&             out   ,
                                          Mat&&            mat   ,
                                          T                dtdh  ,
                                          setter_ref_t     setter)
  {
    if (in_range(in))
    {
      const auto flux_solver = flux_solver_t(mat, dtdh);
            auto patch       = make_patch_iterator(in, dispatch_tag);

      // Shift the iterators to offset the padding, then set the patch data:
      unrolled_for<num_dimensions>([&] (auto dim)
      {
        shift_iterators(in, out, patch, dim);
      });
      *patch = *in;
      __syncthreads();

      unrolled_for<num_dimensions>([&] (auto dim)
      {
        loader_t::load_boundary(in, patch, dim, setter);
      });
      __syncthreads();
      
      // Update states as (for dimension i):
      //  U_i + dt/dh * [F_{i-1/2} - F_{i+1/2}]
      *out = *patch + dtdh * flux_solver.flux_delta(patch, dim);
    }
  }

 private:
  /// Offsets the iterators used by the solver. This will offset the iterators
  /// in the dimension defined by \p dim, and will additionally offset the
  /// patch iterator by the padding amount in the given dimension.
  /// \param[in] in        The input data iterator for solving.
  /// \param[in] out       The output data iterator for solving.
  /// \param[in] patch     The iterator over the patch data.
  /// \param[in] dim       The dimension to offset in.
  /// \tparam    I1        The type of the input and output iterators.
  /// \tparam    I2        The type of the patch iterator.
  /// \tparam    Dim       The type of the offset dimension specifier.
  template <typename I1, typename I2, typename Dim>
  fluidity_host_device static auto
  shift_iterators(I1&& in, I1&& out, I2&& patch, Dim dim)
  {
    const auto shift_amount = flattened_id(dim_off);
    in.shift(shift_amount , dim);
    out.shift(shift_amount, dim);
    patch.shift(thread_id(dim) + padding);
  }

  /// Returns a shared memory multi dimensional iterator over a patch. This
  /// overload is specifically for solving a 1D system.
  /// \param[in] it   The iterator to the start of the global data.
  /// \tparam    IT   The type of the iterator.
  template <typename It>
  fluidity_device_only static auto make_patch_iterator(It&& it, tag_1d_t)
  {
    using state_t    = std::decay_t<decltype(*(it))>;
    using dim_info_t = DimInfoCt<threads_per_block_1d_x + (padding << 1)>;
    return make_multidim_iterator<state_t, dim_info_t>();
  }

  /// Returns a shared memory multi dimensional iterator over a patch. This
  /// overload is specifically for solving a 2D system.
  /// \param[in] it   The iterator to the start of the global data.
  /// \tparam    IT   The type of the iterator.
  template <typename IT>
  fluidity_device_only static auto make_patch_iterator(It&& it, tag_2d_t)
  {
    constexpr auto pad_amount = padding << 1;
    using state_t    = std::decay_t<decltype(*(it))>;
    using dim_info_t = 
      DimInfoCt<
        threads_per_block_2d_x + pad_amount,
        threads_per_block_2d_y + pad_amount>;
    return make_multidim_iterator<state_t, dim_info_t>();
  }

  /// Returns a shared memory multi dimensional iterator over a patch. This
  /// overload is specifically for solving a 2D system.
  /// \param[in] it   The iterator to the start of the global data.
  /// \tparam    IT   The type of the iterator.
  template <typename IT>
  fluidity_device_only static auto make_patch_iterator(It&& it, tag_3d_t)
  {
    constexpr auto pad_amount = padding << 1;
    using state_t    = std::decay_t<decltype(*(it))>;
    using dim_info_t =
      DimInfoCt<
        threads_per_block_3d_x + pad_amount, 
        threads_per_block_3d_y + pad_amount,
        threads_per_block_3d_z + pad_amount>;
    return make_multidim_iterator<state_t, dim_info_t>();
  }
};

}} // namespace fluid::solver


#endif // FLUIDITY_SOLVER_UNSPLIT_SOLVER_HPP