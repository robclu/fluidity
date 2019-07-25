//==--- fluidity/solver/boundary_loader.hpp ---------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  boundary_loader.hpp
/// \brief This file defines functionality for loading data at the boundary.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_BOUNDARY_LOADER_HPP
#define FLUIDITY_SOLVER_BOUNDARY_LOADER_HPP

#include <fluidity/dimension/dimension.hpp>
#include <fluidity/dimension/thread_index.hpp>
#include <fluidity/math/math.hpp>
#include <fluidity/traits/device_traits.hpp>
#include <fluidity/utility/cuda.hpp>

namespace fluid  {
namespace solver {

/// The BoundaryKind enum defines the kind of a boundary.
enum class BoundaryKind {
  transmissive = 0,   //!< Defines a transmissive boundary.
  reflective   = 1    //!< Defines a reflective boundary.   
};

/// The BoundaryIndex enum defines which of the boundaries to set.
enum class BoundaryIndex {
  first  = 0, //!< Defines the index of the first boundary.
  second = 1  //!< Defines the index of the second boundary.
};

/// The BoundarySetter class implements functionality for setting boundary data
/// based on an internal state.
struct BoundarySetter {
  /// Defines the mask to get a BoundaryKind from an integer.
  static constexpr uint16_t mask = 0x0001;

  /// Configures the \p ith boundary for \p dim.
  /// \param[in]  dim   The dimension to configure the boundary setting for.
  /// \param[in]  index The index of the boundary to set (first or second)
  /// \param[in]  kind  The kind of the boundary.
  /// \param[in]  type  The type to set the boundary to.
  /// \tparam     Dim   The type of the dimension.
  template <typename Dim>
  fluidity_host_device constexpr void
  configure(Dim dim, BoundaryIndex index, BoundaryKind kind)
  {
    _configuration ^= (-static_cast<uint8_t>(kind) ^ _configuration)
                    & (1 << bit_index(dim, index));
  }

  /// Overload of operator() to set the value of the \p boundary state to that
  /// of the \p internal state. If the configured type of the boundary for the
  /// \p index boundary in the given dimension is BoundaryKind::reflective, then
  /// the velocity normal to the dimension is flipped.
  /// \param[in] internal   The internal state to use to set the boundary state.
  /// \param[in] boundary   The boundary state to load.
  /// \param[in] dim        The dimension to set the value in.
  /// \param[in] index      The index of the boundary (first or second).
  /// \tparam    StateType  The type of the state.
  /// \tparam    Dim        The type of the dimension.
  template <typename StateType, typename Dim>
  fluidity_host_device void operator()(const StateType& internal,
                                       StateType&       boundary,
                                       Dim              dim     ,
                                       BoundaryIndex    index   ) const
  {
    boundary = internal;
    set_velocity(std::forward<StateType>(boundary), dim, index);
  }

  /// Sets the velocity of the \p boundary by flipping it's value if the kind
  /// of the boundary is BoundaryKind::reflective, otherwise nothing is done.
  /// \param[in] boundary   The boundary state to set the velocity for.
  /// \param[in] dim        The dimension to set the velocity for.
  /// \param[in] index      The index of the boundary (first or second).
  /// \tparam    StateType  The type of the state.
  /// \tparam    Dim        The type of the dimension.
  template <typename StateType, typename Dim>
  fluidity_host_device void
  set_velocity(StateType&& boundary, Dim dim, BoundaryIndex index) const
  {
    if (get_kind(dim, index) == BoundaryKind::reflective)
    {
      boundary.set_velocity(-boundary.velocity(dim), dim);
    }
  }

 private:
  uint16_t _configuration = 0;  //!< Boundary configuration -- all transmissive.
  
  /// Returns the shift amount to move the bits for the boundary type for
  /// dimension \p dim and \p index into the LSB position.
  /// \param[in]  dim     The dimension to get the bit index for.
  /// \param[in]  index   The index (0 = front, 1 = back) of the boundary.
  /// \tparam     Dim     The type of the dimension.
  template <typename Dim>
  fluidity_host_device constexpr std::size_t
  bit_index(Dim dim, BoundaryIndex index) const
  {
    return (static_cast<std::size_t>(dim) << 2) + 
           (static_cast<std::size_t>(dim) << 1);
  }

  /// Returns the type of the boundary for dimension \p with index \p index.
  /// \param[in]  dim     The dimension to get the boundary type for.
  /// \param[in]  index   The index (0 = front, 1 = back) of the boundary.
  /// \tparam     Dim     The type of the dimension.
  template <typename Dim>
  fluidity_host_device constexpr BoundaryKind
  get_kind(Dim dim, BoundaryIndex index) const
  {
    return static_cast<BoundaryKind>(
             (_configuration >> bit_index(dim, index)) & mask);
  }
};

/// The BoundaryLoader struct loads state data into "extra" cells required by a
/// computational block, i.e the additional states required by a cell which is
/// at the start or end of a computational block. For blocks at the start and
/// end of a dimensions the "extra" cells need to be set based on the type of
/// boundary in the computational domain, while for other blocks (i.e if the
/// domain is split into patches) then the values comes from inside the domain.
/// 
/// The loader functionality is thus split into boundary loading and internal
/// loading functionality.
///
/// \tparam Padding The width of the padding (per edge).
template <std::size_t Padding>
struct BoundaryLoader {
  /// Defines the amount of padding used by the loader.
  static constexpr std::size_t padding = Padding;

  /// This sets the boundary elements in a specific dimension for a \p patch,
  /// where the \p patch is an iterator which can iterate over the given
  /// dimension.
  /// 
  /// Loading is done in the following manner:
  /// 
  /// Memory block layout for a single dimension:
  /// 
  /// b = boundary element to set in the patch
  /// s = valid global data to use to set the element
  /// 
  ///    ____________________________          ___________________________
  ///    |           ______         |          |          ______         |
  ///    |           |    |         |          |          |    |         |
  ///    V           V    |         |          |          |    V         V
  /// =======================================================================
  /// | b-n | b-1 | b0 | s0 | s1 | sn | ... | s-n | s-1 | s0 | b0 | b1 | bn |
  /// =======================================================================
  ///          ^              |                      |              ^
  ///          |______________|                      |______________|
  ///          
  /// With the above illustration, the technique used in the code should be
  /// relatively simple to understand.
  /// 
  /// This implementation is only enabled when the iterator is a GPU iterator.
  /// 
  /// \param[in]  data     An iterator to global data.
  /// \param[in]  patch    An iterator to the patch data.
  /// \param[in]  dim      The dimension to set the boundary in.
  /// \param[in]  setter   The object which is used to set the elements.
  /// \tparam     I1       The type of the data iterator.
  /// \tparam     I2       The type of the patch iterator.
  /// \tparam     Dim      The type of the dimension.
  template <typename I1 ,
            typename I2 ,
            typename Dim, traits::gpu_enable_t<I1> = 0>
  static fluidity_host_device void
  load_boundary(I1&& data, I2&& patch, Dim dim, BoundarySetter setter) {
    int global_idx = flattened_id(dim), local_idx = thread_id(dim);
    if (global_idx < padding) {
      constexpr auto bi = BoundaryIndex::first;
      setter(*patch, *patch.offset(-2 * global_idx - 1, dim), dim, bi);
    } else if (local_idx < padding) {
      const auto shift = -2 * local_idx - 1;
      *patch.offset(shift, dim) = *data.offset(shift, dim);
    }

    global_idx = static_cast<int>(data.size(dim)) - global_idx - 1;
    local_idx  = static_cast<int>(block_size(dim)) - local_idx - 1;

    if (global_idx < padding) {
      constexpr auto bi = BoundaryIndex::second;
      setter(*patch, *patch.offset(2 * global_idx + 1, dim), dim, bi);
    } else if (local_idx < padding) {
      const auto shift = 2 * local_idx + 1;
      *patch.offset(shift, dim) = *data.offset(shift, dim);
    }
  }


  template <typename I1 ,
            typename I2 ,
            typename Dim, traits::gpu_enable_t<I1> = 0>
  static fluidity_host_device void
  load_boundary_global(I1&& data, Dim dim, BoundarySetter setter) {
    // Set the near boundary:
    int idx = flattened_id(dim);
    if (idx < padding) {
      constexpr auto bi = BoundaryIndex::first;
      setter(*data, *data.offset(-2 * idx - 1, dim), dim, bi);
    }
    // Set the far boundary:
    idx = static_cast<int>(data.size(dim)) - idx - 1;
    if (idx < padding) {
      constexpr auto bi = BoundaryIndex::second;
      setter(*data, *data.offset(2 * idx + 1, dim), dim, bi);
    }
  }

  /// This sets the boundary elements in a specific dimension for a \p patch,
  /// where the \p patch is an iterator which can iterate over the given
  /// dimension.
  /// 
  /// Loading is done in the following manner:
  /// 
  /// Memory block layout for a single dimension:
  /// 
  /// b = boundary element to set in the patch
  /// s = valid global data to use to set the element
  /// 
  ///    ____________________________          ___________________________
  ///    |           ______         |          |          ______         |
  ///    |           |    |         |          |          |    |         |
  ///    V           V    |         |          |          |    V         V
  /// =======================================================================
  /// | b-n | b-1 | b0 | s0 | s1 | sn | ... | s-n | s-1 | s0 | b0 | b1 | bn |
  /// =======================================================================
  ///          ^              |                      |              ^
  ///          |______________|                      |______________|
  ///          
  /// With the above illustration, the technique used in the code should be
  /// relatively simple to understand.
  /// 
  /// This implementation is only enabled when the iterator is a GPU iterator.
  /// 
  /// \param[in]  data     An iterator to global data.
  /// \param[in]  patch    An iterator to the patch data.
  /// \param[in]  dim      The dimension to set the boundary in.
  /// \param[in]  setter   The object which is used to set the elements.
  /// \tparam     I1       The type of the data iterator.
  /// \tparam     I2       The type of the patch iterator.
  /// \tparam     Dim      The type of the dimension.
  template <typename I1 ,
            typename I2 ,
            typename OW ,
            typename Dim, traits::gpu_enable_t<I1> = 0>
  static fluidity_host_device void 
  load_boundary_unsplit(I1&&           data   ,
                        I2&&           patch  ,
                        OW&&           offwrap,
                        Dim            dim    ,
                        BoundarySetter setter) {
    int global_idx = flattened_id(dim), local_idx = thread_id(dim);
    if (global_idx < padding) {
      constexpr auto bi     = BoundaryIndex::first;
      const auto     shift  = -2 * global_idx - 1;

      setter(*patch, *patch.offset(shift, dim), dim, bi);

      offwrap.offset(shift, dim);
      offwrap.set_as_global();
    } else if (local_idx < padding) {
      const auto shift = -2 * local_idx - 1;

      *patch.offset(shift, dim) = *data.offset(shift, dim);

      offwrap.offset(shift, dim);
      offwrap.set_as_local();
    }

    global_idx = static_cast<int>(data.size(dim)) - global_idx - 1;
    local_idx  = static_cast<int>(block_size(dim)) - local_idx - 1;

    if (global_idx < padding) {
      constexpr auto bi    = BoundaryIndex::second;
      const auto     shift = 2 * global_idx + 1;

      setter(*patch, *patch.offset(shift, dim), dim, bi);

      offwrap.offset(shift, dim);
      offwrap.set_as_global();
    } else if (local_idx < padding) {
      const auto shift = 2 * local_idx + 1;

      *patch.offset(shift, dim) = *data.offset(shift, dim);

      offwrap.offset(shift, dim);
      offwrap.set_as_local();
    }
  }

  /// Loads the corner padding data for a \p patch. This is a little tricky
  /// because the behaviour is different if the corners of the patch are inside
  /// the simulation domain, and hence the data needs to be set from the \p data
  /// iterator.
  ///
  /// Alternatively, the patch corner may be outside of the simulation domain,
  /// in which case it is a boundary condition, and needs to be set from the
  /// other loaded patch data to which the boundary conditions have already
  /// been applied. 
  ///
  /// Lastly, there is the case that the corner of the patch which needs to be
  /// loaded is not actually a corner becase the size of the simulation domain
  /// is not a multiple of the patch size in the given dimension. For example,
  /// if the computational domain is 100 elements in the dimension, and the
  /// block size is 16, thread 3 (the global boundary element) will need to have
  /// the same loading behaviour as if it was thread 15.
  /// 
  ///
  /// \note There is one case which is not handled here, and that is the case
  /// that the last block in a dimension has the same number of elements as the
  /// padding width, in which case the thread needs to load the corner data for
  /// two corners.
  ///
  /// Handing all these cases is tricky.
  ///
  /// \param[in] data  An iterator to the global data.
  /// \param[in] patch An iterator over the patch data to load corners for.
  /// \tparam    D     The number of dimensions for the iterators.
  /// \tparam    I1    The type of the data iterator.
  /// \tparam    I2    The type of the patch iterator.
  template <std::size_t D ,
            typename    I1,
            typename    I2, traits::gpu_enable_t<I1> = 0>
  fluidity_host_device static void load_corners(I1&& data, I2&& patch) {
    auto outer_it = patch;
    auto inner_it = data;

    int in = 0, out = 0;

      auto print_vec = [] (auto v, int b)
       {
 //        if ((thread_id(0) == 0 || thread_id(0) == block_size(0) - 1) &&
 //            (thread_id(1) == 0 || thread_id(1) == block_size(1) - 1))
         if (true)
         {
           printf("TX, TY, BX, BY: { %03lu, %03lu } : {%03lu, %03lu }, { %03i, %03i } : { %03i }\n",
             thread_id(0), thread_id(1), block_id(0), block_id(1), v[0], v[1], b);
         }
       };

    Array<int, D> a{0}, b{0};

    using velocity_t  = std::decay_t<decltype(patch->velocity(0))>;
    using vel_array_t = Array<velocity_t, D>;
    vel_array_t velocities{0};

    unrolled_for<D>([&] (auto d) {
      constexpr auto dim = std::size_t{d};
      auto tid = thread_id(dim) , ftid = flattened_id(dim);
      auto bs  = block_size(dim), gs   = data.size(dim);

      auto sign        = tid < padding ? int{-1} : int{1};
      auto shift_inner = std::min(tid, bs - tid - 1);
      auto shift_outer = std::min(ftid, ftid < gs ? gs - ftid - 1 : gs + 100);

      auto out_prev = out;
      if (shift_outer < padding)
      {
        out++;
        //print_vec(b, dim);
      }
      else if (shift_inner < padding)
      {
        in++;
      }
/*
      if (shift_inner >= padding - 1 && !last)
      {
        //outer++;
        inner = false;
        //inner = false;
      }
*/

/*
        if (block_id(dim) == grid_size(dim) / block_size(dim) - 1)
        {
          b[0] = in;
          b[1] = out;
          //print_vec(b, dim);
        }
*/
      //if (inner)
      if (in + out)
      {
        auto shift_amt   = out_prev < out ? shift_outer : shift_inner;
        int shift_amount = 2 * shift_amt * sign + sign;
        a[dim] = shift_amount;

        inner_it.shift(shift_amount, dim);
        outer_it.shift(shift_amount, dim);

        if (std::abs(shift_amount) < 2 * padding + 1)
        {
          velocities[dim] = patch.offset(shift_amount, dim)->velocity(dim);
        }
        //print_vec(a, dim);
      }
    });
    //if (outer && inner)
    if (out + in == 2 && out > 0)
    {
      //print_vec(a, 0);
      *outer_it = *patch;

      // Need to set the velocity boundary conditions at the boundaries:

      unrolled_for<D>([&] (auto d)
      {
        constexpr auto dim = std::size_t{d};
        outer_it->set_velocity(velocities[dim], dim);
      });

      //print_vec(a, -10);
    }
    else if (in == 2)
//    else if (inner)
    {
//            print_vec(a, 1);
      *outer_it = *inner_it;
     //print_vec(a, 22);
    }
  }

  template <typename IT, traits::gpu_enable_t<IT> = 0>
  static fluidity_host_device void 
  load_global_boundaries(IT&& data, const BoundarySetter& setter)
  {
    auto corner_it = data;
    auto corner    = *data;
    auto is_corner = 0;

    constexpr auto dims = std::decay_t<IT>::dimensions;

    unrolled_for<dims>([&] (auto dim)
    {
      auto idx = flattened_id(dim), elements = data.size(dim) - (padding << 1);

      auto sign  = idx < (elements >> 1) ? int{-1} : int{1};
      auto shift = 
        std::min(idx, idx < elements ? elements - idx - 1 : elements << 2);

      if (shift < padding)
      {
        int amount = 2 * shift * sign + sign;
        auto bi    = sign < 0 ? BoundaryIndex::first : BoundaryIndex::second;
        setter(*data, *data.offset(amount, dim), dim, bi);
        corner.set_velocity(data.offset(amount, dim)->velocity(dim), dim);
        corner_it.shift(amount, dim);
        is_corner++;
      }
    });

    // One of the global corners, and we need to set that data too.
    if (is_corner == 2) { *corner_it = corner; }
  }

  template <typename I1, typename I2, traits::gpu_enable_t<I1> = 0>
  static fluidity_host_device void load_patch_boundaries(I1&& data, I2&& patch)
  {
    auto patch_corner_it = patch;
    auto data_corner_it  = data;
    auto is_corner       = 0;

    constexpr auto dims = std::decay_t<I1>::dimensions;

    unrolled_for<dims>([&] (auto dim)
    {
      auto idx      = thread_id(dim);
      auto elements = std::min(
        block_size(dim), data.size(dim) - (padding << 1) - block_id(dim) * block_size(dim));

      auto sign  = idx < (elements >> 1) ? int{-1} : int{1};
      auto shift = 
        std::min(idx, idx < elements ? elements - idx - 1 : elements << 2);

      if (shift < padding)
      {
        int amount = 2 * shift * sign + sign;
        // Set the non-corner data:
        *patch.offset(amount, dim) = *data.offset(amount, dim);

        // Move the corner data iterators:
        data_corner_it.shift(amount, dim);
        patch_corner_it.shift(amount, dim);
        is_corner++;
      }
    });

    // Set the data in the corner of the patch:
    if (is_corner == 2) { *patch_corner_it = *data_corner_it; }
  }

  /// This sets the boundary elements in a specific dimension for a \p patch,
  /// where the \p patch is an iterator which can iterate over the given
  /// dimension.
  /// 
  /// Loading is done in the following manner:
  /// 
  /// Memory block layout for a single dimension:
  /// 
  /// b = boundary element to set in the patch
  /// s = valid global data to use to set the element
  /// 
  ///    ____________________________          ___________________________
  ///    |           ______         |          |          ______         |
  ///    |           |    |         |          |          |    |         |
  ///    V           V    |         |          |          |    V         V
  /// =======================================================================
  /// | b-n | b-1 | b0 | s0 | s1 | sn | ... | s-n | s-1 | s0 | b0 | b1 | bn |
  /// =======================================================================
  ///          ^              |                      |              ^
  ///          |______________|                      |______________|
  ///          
  /// With the above illustration, the technique used in the code should be
  /// relatively simple to understand.
  /// 
  /// This implementation is only enabled when the iterator is a CPU iterator.
  /// 
  /// \param[in]  data     An iterator to global data.
  /// \param[in]  patch    An iterator to the patch data.
  /// \param[in]  dim      The dimension to set the boundary in.
  /// \param[in]  setter   The object which is used to set the elements.
  /// \tparam     I1       The type of the data iterator.
  /// \tparam     I2       The type of the patch iterator.
  /// \tparam     Dim      The type of the dimension.
  template <typename I1 ,
            typename I2 ,
            typename Dim, traits::cpu_enable_t<I1> = 0>
  static fluidity_host_device void
  load_boundary(I1&& data, I2&& patch, Dim dim, BoundarySetter setter)
  {
    /*
     * TODO: Implement ...
     */
  }
};

}} // namespace fluid::solver

#endif // FLUIDITY_SOLVER_BOUNDARY_LOADER_HPP
