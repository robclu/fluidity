//==--- fluidity/solver/multimaterial_split_solver.hpp ----- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  multimaterial_split_solver.hpp
/// \brief This file defines implementations of a split solver for multi
///        materials which uses level sets to define the regions for the
///        different materials.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_MULTIMATERIAL_SPLIT_SOLVER_HPP
#define FLUIDITY_SOLVER_MULTIMATERIAL_SPLIT_SOLVER_HPP

//#include "solver_utilities.hpp"
#include "cuda/split_solver.cuh"
#include <fluidity/dimension/dimension.hpp>
#include <fluidity/execution/execution_policy.hpp>
#include <fluidity/iterator/multidim_iterator.hpp>
#include <fluidity/utility/cuda.hpp>
#include <fluidity/utility/number.hpp>

namespace fluid  {
namespace solver {

/// The SplitSolver class defines an implementation of a solver which updates
/// states using a dimensionally split method. It can be specialized for
/// systems of different dimension. This implementation is defaulted for the 1D
/// case.
/// \tparam FluxSolver The method to use for solving the face fluxes.
/// \tparam Loader     The implementation for boundary loading.
/// \tparam Dimensions The number of dimensions to solve over.
template <typename FluxSolver, typename Loader, typename Dims = Num<1>>
struct MultiMaterialSplitSolver {
 private:
  /// Defines the type of the flux solver.
  using flux_solver_t = std::decay_t<FluxSolver>;
  /// Defines the type of the loader for the data.
  using loader_t      = std::decay_t<Loader>;
  /// Defines the type of the boundary setter.
  using setter_t      = BoundarySetter;
  /// Defines a reference type to the boundary setter.
  using setter_ref_t  = const BoundarySetter&;
  /// Defines the data type used to identify materials.
  using material_id_t = uint8_t;

  /// Defines the number of dimensions to solve over.
  static constexpr auto num_dimensions  = std::size_t{Dims()};
  /// Defines the amount of padding in the data loader, on one side of a dim.
  static constexpr auto padding         = loader_t::padding;
  /// Defines the amount of padding on both sides of a domain.
  static constexpr auto padding_both    = (padding << 1);
  /// Defines the dispatch tag for dimension overloading.
  static constexpr auto dispatch_tag    = dim_dispatch_tag<num_dimensions>;

#if defined(FLUIDITY_CUDA_AVAILABLE)
  dim3 _thread_sizes;  //!< The number of threads in each dim of a block.
  dim3 _block_sizes;   //!< The number of blocks in each dim.
#else
  Dim3   _thread_sizes;
  Dim3   _block_sizes;
#endif // FLUIDITY_CUDA_AVAILBLE

 public:
  /// Creates the split solver, initializing the number of threads and blocks
  /// based on the dimensionality of the iterator.
  /// \param[in] it The iterator to use for solving.
  /// \tparam    It The type of the iteraotr.
  template <typename It>
  SplitSolver(It&& it)
  : _thread_sizes{get_thread_sizes(it)}, 
    _block_sizes{get_block_sizes(it, _thread_sizes)} {}

  /// Updater function for updating the simulation. This overload is only
  /// enabled when the input and output iterators are for GPU execution.
  /// This simply forwards all the arguments onto the cuda implementation of the
  /// solving function.
  /// 
  /// \param[in] in    The input data to use to update.
  /// \param[in] out   The output data to write to after updating.
  /// \param[in] mat   The material for the system.
  /// \param[in] dtdh  Scaling factor for the update.
  /// \tparam    It    The type of the multi dimensional iterator.
  /// \tparam    Mat   The type of the material for the system.
  /// \tparam    T     The type of the scaling factor.
  template <typename It, typename Mat, typename T, exec::gpu_enable_t<It> = 0>
  void solve(It&& in, It&& out, Mat&& mat, T dtdh, setter_ref_t setter) const 
  {
    detail::cuda::solve_impl_split(*this                 ,
                                   std::forward<It>(in)  ,
                                   std::forward<It>(out) ,
                                   std::forward<Mat>(mat),
                                   dtdh                  ,
                                   setter                );
  }

  /// Returns the number of threads per block for the solver.
  auto thread_sizes() const
  {
    return _thread_sizes;
  }

  /// Returns the dimension information for the blocks to solve.
  auto block_sizes() const
  {
    return _block_sizes;
  }

  /// Overlaod of the call operator to invoke a pass of solving on the input and
  /// output data iterators for a specific dimension which is defined by Value.
  /// The data from the \p in iterator is used to compute the update which is
  /// then written to the \p out iterator.
  /// \param[in] in     The input multi dimensional iterator over state data.
  /// \param[in] out    The output multi dimensional iterator over state data.
  /// \param[in] mats   The materials for the system.
  /// \tparam    It     The type of the iterator.
  /// \tparam    Mats   The type of the materials for the system.
  /// \tparam    T      The data type for the scaling factor.
  /// \tparam    Value  The value which defines the dimension for the pass.
  template <typename It, typename Mat, typename T, typename Dim>
  fluidity_device_only static void invoke(It&&             in    ,
                                          It&&             out   ,
                                          Mats&&           mats  ,
                                          T                dtdh  ,
                                          setter_ref_t     setter,
                                          Dim              dim   )
  {
    if (in_range(in))
    {
      const auto flux_solver = flux_solver_t(mat, dtdh);
            auto patch       = make_patch_iterator(in, dispatch_tag, dim);
            auto materials   = make_material_iterator(dispatch_tag, dim);

      // Shift the iterators to offset the padding, then set the patch data:
      unrolled_for<num_dimensions>([&] (auto dim_off)
      {
        shift_iterators(in, out, mats, materials, patch, dim_off, dim);

        // Also need to shift the material iterators.
        shift_material_iterators(mats, materials, dim_off, dim);
      });
      *patch = *in;

      // Set which material each cell belongs to.
      uint8_t i  = 0;
      *materials = 0;
      for (const auto& material : mats)
      {
        if (material.is_inside())
          *materials = i;
      }

      __syncthreads();
      loader_t::load_boundary(in, patch, dim, setter);

      // Update states as (for dimension i):
      //  U_i + dt/dh * [F_{i-1/2} - F_{i+1/2}]
      *out = *patch + dtdh * flux_solver.flux_delta(patch, dim);
    }
  }

 private:
  /// Evolves the level sets for the 



  /// Offsets the iterators used by the solver. This will offset the iterators
  /// in the dimension defined by \p dim_off, and will additionally offset the
  /// patch iterator by the padding amount if the \p dim_solve is the same as
  /// the offset dim \p dim_off.
  /// \param[in] in        The input data iterator for solving.
  /// \param[in] out       The output data iterator for solving.
  /// \param[in] patch     The iterator over the patch data.
  /// \param[in] dim_off   The dimension to offset in.
  /// \param[in] dim_solve The dimension to solve in.
  /// \tparam    I1        The type of the input and output iterators.
  /// \tparam    I2        The type of the patch iterator.
  /// \tparam    ODim      The type of the offset dimension specifier.
  /// \tparam    SDim      The type of the solve dimension specifier.
  template <typename I1, typename I2, typename I3&&, typename ODim, typename SDim>
  fluidity_host_device static auto
  shift_iterators(I1&& in       ,
                  I1&& out      ,
                  I2&& patch    ,
                  I3&& mats     ,
                  ODim dim_off  ,
                  SDim dim_solve)
  {
    const auto in_out_shift = flattened_id(dim_off);
    in.shift(in_out_shift , dim_off);
    out.shift(in_out_shift, dim_off);

    const auto shift =
      thread_id(dim_off) + (dim_solve == dim_off ? padding : 0);
    patch.shift(shift, dim_off);
    mats.shift(shift, dim_off);
  }

  /// Offsets the iterators used by the solver. This will offset the iterators
  /// in the dimension defined by \p dim_off, and will additionally offset the
  /// patch iterator by the padding amount if the \p dim_solve is the same as
  /// the offset dim \p dim_off.
  /// \param[in] in        The input data iterator for solving.
  /// \param[in] out       The output data iterator for solving.
  /// \param[in] patch     The iterator over the patch data.
  /// \param[in] dim_off   The dimension to offset in.
  /// \param[in] dim_solve The dimension to solve in.
  /// \tparam    I1        The type of the input and output iterators.
  /// \tparam    I2        The type of the patch iterator.
  /// \tparam    ODim      The type of the offset dimension specifier.
  /// \tparam    SDim      The type of the solve dimension specifier.
  template <typename I1, typename I2, typename I3&&, typename ODim, typename SDim>
  fluidity_host_device static auto
  shift_material_iterators(I1&& mats, I2&& matids, ODim dim_off, SDim dim_solve)
  {
    const auto levelset_shift = flattened_id(dim_off);
    // Shift each of the material level sets.

    const auto shift =
      thread_id(dim_off) + (dim_solve == dim_off ? padding : 0);
    matids.shift(shift, dim_off);
  }


  /// Returns a shared memory multi dimensional iterator over a patch. 
  /// \param[in] it       The iterator to the start of the global data.
  /// \tparam    Iterator The type of the iterator.
  /// \tparam    DS       The type which defines the solving dimension.
  template <typename It, typename DS>
  fluidity_device_only static auto make_patch_iterator(It&& it, tag_1d_t, DS)
  {
    using state_t    = std::decay_t<decltype(*(it))>;
    using dim_info_t = DimInfoCt<threads_per_block_1d_x + (padding << 1)>;
    return make_multidim_iterator<state_t, dim_info_t>();
  }

  /// Returns a shared memory multi dimensional iterator which specifies which
  /// material each cell belongs to. This specialization is for a 1D system.
  /// \tparam DS The type which defines the solving dimension.
  template <typename DS>
  fluidity_device_only static auto make_material_iterator(tag_1d_t, DS)
  {
    using dim_info_t = DimInfoCt<threads_per_block_1d_x + (padding << 1)>;
    return make_multidim_iterator<material_id_t, dim_info_t>();
  }

  /// Returns a shared memory multi dimensional iterator over a patch. This
  /// overload is called for a 2D system when solving in the x direction.
  /// \param[in] it       The iterator to the start of the global data.
  /// \tparam    Iterator The type of the iterator.
  /// \tparam    DS       The type which defines the solving dimension.
  template <typename It, typename DS>
  fluidity_device_only static auto make_patch_iterator(It&& it, tag_2d_t, DS)
  {
    constexpr auto pad_amount = padding << 1;
    using state_t    = std::decay_t<decltype(*(it))>;
    using dim_info_t = 
      DimInfoCt<
        threads_per_block_2d_x + (DS::value == dimx_t::value ? pad_amount : 0),
        threads_per_block_2d_y + (DS::value == dimy_t::value ? pad_amount : 0)>;
    return make_multidim_iterator<state_t, dim_info_t>();
  }

  /// Returns a shared memory multi dimensional iterator which specifies which
  /// material each cell belongs to. This specialization is for a 2D system.
  /// \tparam    DS       The type which defines the solving dimension.
  template <typename DS>
  fluidity_device_only static auto make_material_iterator(tag_2d_t, DS)
  {
    constexpr auto pad_amount = padding << 1;
    using dim_info_t = 
      DimInfoCt<
        threads_per_block_2d_x + (DS::value == dimx_t::value ? pad_amount : 0),
        threads_per_block_2d_y + (DS::value == dimy_t::value ? pad_amount : 0)>;
    return make_multidim_iterator<material_id_t, dim_info_t>();
  }

  /// Returns a shared memory multi dimensional iterator over a patch. This
  /// overload is called for a 3D system when solving in the x direction.
  /// \param[in] it       The iterator to the start of the global data.
  /// \tparam    Iterator The type of the iterator.
  /// \tparam    DS       The type which defines the solving dimension.
  template <typename It, typename DS>
  fluidity_device_only static auto make_patch_iterator(It&& it, tag_3d_t, DS)
  {
    constexpr auto pad_amount = padding << 1;
    using state_t    = std::decay_t<decltype(*(it))>;
    using dim_info_t = 
      DimInfoCt<
        threads_per_block_3d_x + (DS::value == dimx_t::value ? pad_amount : 0), 
        threads_per_block_3d_y + (DS::value == dimy_t::value ? pad_amount : 0),
        threads_per_block_3d_z + (DS::value == dimz_t::value ? pad_amount : 0)>;
    return make_multidim_iterator<state_t, dim_info_t>();
  }

  /// Returns a shared memory multi dimensional iterator for identifying which
  /// material a cell belongs to. This specialization is for a 3D system.
  /// \tparam    DS       The type which defines the solving dimension.
  template <typename DS>
  fluidity_device_only static auto make_material_iterator(tag_3d_t, DS)
  {
    constexpr auto pad_amount = padding << 1;
    using dim_info_t = 
      DimInfoCt<
        threads_per_block_3d_x + (DS::value == dimx_t::value ? pad_amount : 0), 
        threads_per_block_3d_y + (DS::value == dimy_t::value ? pad_amount : 0),
        threads_per_block_3d_z + (DS::value == dimz_t::value ? pad_amount : 0)>;
    return make_multidim_iterator<material_id_t, dim_info_t>();
  }


};

}} // namespace fluid::solver

#endif // FLUIDITY_SOLVER_MULTIMATERIAL_SPLIT_SOLVER_HPP