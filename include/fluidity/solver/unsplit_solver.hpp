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

#include "flux_solver.hpp"
#include "solver_utilities.hpp"
#include "cuda/unsplit_solver.cuh"
#include <fluidity/execution/execution_policy.hpp>

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
  /// Defines the type of the reconstructor.
  using recon_t       = typename flux_solver_t::reconstructor_t;
  /// Defines the type of the limiter.
  using limiter_t     = typename recon_t::limiter_t;

  /// Define the type of the flux solver for the perpendicular fluxes.


  /// Defines the number of dimensions to solve over.
  static constexpr auto num_dimensions = std::size_t{Dims()};
  /// Defines the amount of padding in the data loader.
  static constexpr auto padding        = loader_t::padding;
  /// Defines the dispatch tag for dimension overloading.
  static constexpr auto dispatch_tag   = dim_dispatch_tag<num_dimensions>;
  /// Defines the width of the stencil.
  static constexpr auto width          = recon_t::width;

#if defined(FLUIDITY_CUDA_AVAILABLE)
  dim3 _thread_sizes;  //!< The number of threads in each dim of a block.
  dim3 _block_sizes;   //!< The number of blocks in each dim.
#else
  Dim3   _thread_sizes;
  Dim3   _block_sizes;
#endif // FLUIDITY_CUDA_AVAILBLE

 public:
  /// Creates the unsplit solver.
  /// \param[in] it The iterator over the computational space to solve.
  /// \tparam    It The type of the iterator.
  template <typename It>
  UnsplitSolver(It&& it) 
  : _thread_sizes{get_thread_sizes(it, padding << 1)}, 
    _block_sizes{get_block_sizes(it, _thread_sizes, padding << 1)} {}

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
    detail::cuda::solve_impl_unsplit(*this, 
                                     std::forward<It>(in)  ,
                                     std::forward<It>(out) ,
                                     std::forward<Mat>(mat),
                                     dtdh                  ,
                                     setter                );
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
  template <typename It, typename Mat, typename T>
  fluidity_device_only static void invoke(It&&             in    ,
                                          It&&             out   ,
                                          Mat&&            mat   ,
                                          T                dtdh  ,
                                          setter_ref_t     setter)
  {
      auto print_vec = [] (auto v, auto d, auto p)
      {
        //if (flattened_block_id(0) == 0 && thread_id(1) == 0)
        if (v[0] != 0.0 || v[1] != 0.0 || v[2] != 0.0 || v[3] != 0.0)
        {
          printf("TX, TY, BX, BY, GX, GY : { %03lu, %03lu } : { %03lu, %03lu }, {%03lu, %03lu }, { %4.4f, %4.4f, %4.4f, %4.4f }\n",
            thread_id(0), thread_id(1), 
            block_id(0), block_id(1),
            flattened_id(0), flattened_id(1),
            v[0], v[1], v[2], v[3]);
        }
      };

      auto pool = [&] (auto it, auto&& p)
      {
        auto r = std::decay_t<decltype(p(*it))>();
        for (auto off_1 : range(-1, 2))
        {
          auto i = it.offset(off_1, 0);
          for (auto off_2 : range(-1, 2))
          {
            r += p(*i.offset(off_2, 1));
          }
        }
        return r;
      };

    using perp_flux_solver_t = 
      FaceFlux<
        recon::BasicReconstructor<limiter_t> ,
        typename flux_solver_t::flux_method_t,
        std::decay_t<decltype(mat)>          >;

    if (in_range(in, padding << 1))
    {
      const auto flux_solver  = flux_solver_t(mat, dtdh);
      const auto pflux_solver = perp_flux_solver_t(mat, dtdh); 
            auto patch        = make_patch_iterator(in, dispatch_tag);
//            auto flux            = make_patch_iterator(in, dispatch_tag);

      //using flux_sum_t = std::decay_t<decltype(*flux)>;
      using flux_sum_t = std::decay_t<decltype(*patch)>;
      auto flux_sum    = flux_sum_t(0.0, 0.0, 0.0, 0.0);

      constexpr auto f_off = int{1};
      constexpr auto b_off = -f_off;

      // Shift the iterators to offset the padding, then set the patch data:
      unrolled_for<num_dimensions>([&] (auto dim)
      {
        const auto shift_global = flattened_id(dim) + padding;
        const auto shift_local  = thread_id(dim)    + padding;
        in.shift(shift_global, dim);
        out.shift(shift_global, dim);
        patch.shift(shift_local, dim);
//        flux.shift(shift_local, dim);
      });
     *patch = *in;

      // Load the boundary patch data from the global data:
      loader_t::load_patch_boundaries(in, patch);
      __syncthreads();
/*
      unrolled_for<num_dimensions>([&] (auto dim)
      {
        loader_t::load_boundary(in, patch, dim, setter);
      });
      loader_t::template load_corners<num_dimensions>(in, patch);
      __syncthreads();
*/

      // Compute the flux contibution from each dimension:
      unrolled_for<num_dimensions - 1>([&] (auto d)
      {
        constexpr auto dim = std::size_t{0};
//        constexpr auto dim = std::size_t{1};
        // For each dimension, compute the flux difference in the dimensions
        // perpendicular, and store those in the shared flux memory.
        unrolled_for<num_dimensions - 1>([&] (auto i)
        {
       //   constexpr auto dim  = std::size_t{d};
          constexpr auto pdim = (dim + i + std::size_t{1}) % num_dimensions;

          //flux.shift(-f_off, dim);
          //patch.shift(-f_off, dim);
          //__syncthreads();
          // Move the iterators forward in the __dim__ (i.e solving) dimension,
          // and then solve for the flux difference in that offset cell in the
          // perpendicular dimension.

          auto flux_c = pflux_solver.flux_delta(patch, pdim);
          auto flux_b = pflux_solver.flux_delta(patch.offset(b_off, dim), pdim);
          auto flux_f = pflux_solver.flux_delta(patch.offset(f_off, dim), pdim);

          //*flux = perpflux_solver.flux_delta(patch, pdim);
//          *flux.shift(f_off, dim) 
//            = perpflux_solver.flux_delta(patch.shift(f_off, dim), pdim);

          //*flux = perpflux_solver.flux_delta(patch, pdim);

//          flux.shift(b_off, dim);
//          patch.shift(b_off, dim);
          // Need to load 2 * width more fluxes (width fluxes at the start
          // because of the shift above, and width fluxes in the padded region
          // at the start of the dim dimension).
//          if (thread_id(dim) < (f_off << 1))
//          {
            //*flux.offset(b_off, dim) = 
            //  perpflux_olver.flux_delta(patch.offset(b_off, dim), pdim);
//          }

          // Need to sync here since we are now going to use the flux deltas on
          // either side (in the dim direction) of this cell.
          //__syncthreads();
          flux_sum = flux_solver.flux_delta(patch, dim);
/*
          flux_sum += 
//            pool(patch, [&] (auto& x) { return x; });

            flux_solver.backward(patch, dim, [&] (auto& l, auto& r)
            {
              l -= -0.5 * dtdh * flux_b;
              r -= -0.5 * dtdh * flux_c;
            })
          -
            flux_solver.forward(patch, dim, [&] (auto& l, auto& r)
            {
              l -= -0.5 * dtdh * flux_c;
              r -= -0.5 * dtdh * flux_f;
            });
*/
//              print_vec(flux_sum, std::size_t{dim}, std::size_t{pdim});
          if (flux_sum[0] != 0.000 && flux_sum[1] != 0.00)
            print_vec(flux_sum, std::size_t{0}, std::size_t{dim});
        });

        // Can't start on the next dimension until this one is done ...
        //__syncthreads();
      });
      //__syncthreads();
      // Update states as (for dimension i):
      //  U_i + dt/dh * lambda + sum_{i=0..D} [F_{i-1/2} - F_{i+1/2}]
      //*out = *patch + dtdh * flux_sum;
      *out = flux_sum;
      //out->set_density(1.0);

      //patch->set_density(4.0);
//      *out = pool(patch, [] (auto& v) { return v; });

      // Load the boundary data for the output iterator, so that for the next
      // iteration the boundaries are loaded:
      loader_t::load_global_boundaries(out, setter);

//      if (flattened_id(0) == 0 && flattened_id(1) == 0)
//        printf("\n\n");
    }
  }

 private:
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
  /// \tparam    It   The type of the iterator.
  template <typename It>
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
  /// \tparam    It   The type of the iterator.
  template <typename It>
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