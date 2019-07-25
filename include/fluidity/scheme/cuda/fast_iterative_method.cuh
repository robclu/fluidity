//==--- fluidity/scheme/cuda/fast_iterative_method.cuh ----- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  fast_iterative_method.hpp
/// \brief This file provides a cuda implementation of the fast iterative
///        method.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_CUDA_FAST_ITERATIVE_METHOD_CUH
#define FLUIDITY_SCHEME_CUDA_FAST_ITERATIVE_METHOD_CUH

#include <fluidity/execution/execution_policy.hpp>
#include <fluidity/math/math.hpp>
#include <fluidity/scheme/schemes/godunov_upwind_scheme.hpp>
#include <fluidity/traits/iterator_traits.hpp>
#include <fluidity/traits/tensor_traits.hpp>
#include <fluidity/utility/constants.hpp>
#include <fluidity/utility/portability.hpp>
#include <array>
#include <limits>

namespace fluid  {
namespace scheme {
namespace cuda   {

//==--- [Kernels] ----------------------------------------------------------==//

namespace kernel {

/// Initializes the data for the fast iterative method. The method is
/// initialized by looking at the value of the input data, and if it is less
/// than \p width, the value of the input is set to the output, otherwise the
/// output value is set to the max value for the type.
///
/// \param[in] in_it    The input data iterator.
/// \param[in] out_it   The output data iterator.
/// \param[in] width    The width of the band which defines the source.
/// \tparam    Iterator The type of the input and output iterators.
/// \tparam    T        The type of the width information.
template <typename Iterator, typename T>
fluidity_global auto fast_iterative_init(
  Iterator in_it ,
  Iterator out_it,
  T        width
) -> void {
  using iter_t = std::decay_t<Iterator>;
  constexpr auto max_v = std::numeric_limits<T>::max();

  // Offset the input and output iterators to their respective places ...
  unrolled_for<iter_t::dimensions>([&] (auto dim) {
    const auto flat_idx = flattened_id(dim);
    in_it.shift(flat_idx, dim);
    out_it.shift(flat_idx, dim);
  });

  // The source node is the smallest cell next to the interface ...
  bool source = false;
  for (auto dim : range(iter_t::dimensions)) {
    const auto this_cell  = *in_it;
    auto       other_cell = *in_it.offset(1, dim);
    
    if (math::signum(this_cell) != math::signum(other_cell) &&
        std::abs(this_cell) < std::abs(other_cell)) {
      source = true;
      break;
    }

    other_cell = *in_it.offset(-1, dim);
    if (math::signum(this_cell) != math::signum(other_cell) &&
        std::abs(this_cell) < std::abs(other_cell)) {
      source = true;
      break;
    }
  }

  *out_it = source ? *in_it : max_v;
}

// The gradient difference due to the sign change.
fluidity_device_only double grad_diff;

/// Sets the sign of the \p out_it output data to have the same sign as
/// the \p in_it input iterator.
/// \param[in] in_it    An input iterator to get the sign from.
/// \param[in] out_it   An iterator to the output data to set the sign for.
/// \tparam    Iterator The type of the in and output iterators.
template <typename Iterator, typename T>
fluidity_global auto set_signs(Iterator in_it, Iterator out_it, T dh)
-> void {
  using iter_t = std::decay_t<Iterator>;
  constexpr auto epsilon = T{1e-2};

  if (!in_range(in_it)) {
    return;
  }

  unrolled_for<iter_t::dimensions>([&] (auto dim) {
    const auto flat_idx = flattened_id(dim);
    in_it.shift(flat_idx, dim);
    out_it.shift(flat_idx, dim);
  });
  *out_it = math::signum(*in_it) * std::abs(*out_it);
}

/// When resetting the sign of the signed difference function, error may be
/// introduced due to the symmetry of the fast iterative method. For example,
/// when using the following initial data:
///
/// \code{bash}
///   | 0.009656 | 0.003581 | -0.008271 | -0.01579 |
/// \endcode
///
/// After solving with the fast iterative method, if cell ```0.003581``` is the
/// source node, the updated values may be:
///
/// \code{bash}
///   | 0.01108 | 0.003581 | 0.01108 | 0.01858 |
///                   |           |
///                   -------------
///                       DELTA = 0.0075
/// \endcode
///
/// where the gradients are now all correct, (and the same) however, after
/// resetting the signs, the data becomes:
///
/// \code{bash}
///   | 0.01108 | 0.003581 | -0.01108 | -0.01858 |
///                  |            |
///                  --------------
///                      ERROR - delta changes (!= 0.0075)
/// \endcode
///
/// and there is an error due to the sign change. All the negative data needs to
/// be shifted by the error so that the gradient is again the same and correct.
///
/// This function computes the gradient so that it can then be used to shift all
/// the appropriate data.
///
/// \param[in] in_it    An input iterator to get the sign from.
/// \tparam    Iterator The type of the in and output iterators.
template <typename Iterator, typename T>
fluidity_global auto compute_gradient_error(Iterator in_it, T dh)
-> void {
  using iter_t = std::decay_t<Iterator>;
  constexpr auto epsilon = T{1e-7};

  if (!in_range(in_it)) {
    return;
  }

  unrolled_for<iter_t::dimensions>([&] (auto dim) {
    in_it.shift(flattened_id(dim), dim);
  });

  for (auto dim : range(iter_t::dimensions)) {
    const auto flat_idx = flattened_id(dim);
    const auto err_cell = 
      flat_idx != 0                    &&
      flat_idx < (in_it.size(dim) - 1) && 
      std::abs(std::abs(in_it.forward_diff(dim)) - dh) > epsilon;
   
    //{
    //if (std::abs(std::abs(in_it.forward_diff(dim)) - dh) > epsilon) {
    if (err_cell) {
      const auto a = std::abs(*in_it);
      const auto b = std::abs(*in_it.offset(1, dim));
      const auto source = a < b ? *in_it                : *in_it.offset(1, dim);
      const auto other  = a < b ? *in_it.offset(1, dim) : *in_it;

      grad_diff = other - source + dh;
      return;
    }
  }
}

/// Applies the computed gradient errors to the appropriate cells. See
/// compute_gradient_error for more information.
/// \param[in] in_out_it  The input and output iterator to fix.
/// \tparam    Iterator   The type of the in and output iterators.
template <typename Iterator, typename T>
fluidity_global auto fix_gradient_error(Iterator in_out_it, T dh)
-> void {
  using iter_t = std::decay_t<Iterator>;

  if (!in_range(in_out_it)) {
    return;
  }

  unrolled_for<iter_t::dimensions>([&] (auto dim) {
    in_out_it.shift(flattened_id(dim), dim);
  });

  // Any cells inside, but by more than the resolution. 
  if (*in_out_it < -dh) {
    *in_out_it -= grad_diff;
    return;
  }

  // Handle case that the second cell inside the levelset is less than the
  // resolution.
  for (auto dim : range(iter_t::dimensions)) {
    if (levelset::inside(in_out_it)                  && 
        levelset::inside(in_out_it.offset(1 , dim))  &&
        levelset::inside(in_out_it.offset(-1, dim))) {
      *in_out_it -= grad_diff;
      return;
    }
  }
}

//==--- [Broadcast sums] ---------------------------------------------------==//

/// Performs a reduction sum over the entire warp using the value \p val for
/// each of the threads. Each thread gets the result of the sum.
/// \param[in] val    The value to add to the reduction for this thread.
/// \tparam    T      The type of the value.
template <typename T>
fluidity_device_only T warp_broadcast_sum(T val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xffffffff, val, offset, warpSize);
  }
  return val;
};

/// Performs a reduction sum over the entire block using the value \p val for
/// each of the threads. The result of the sum is returned to all threads in the
/// block.
/// \param[in] val    The value to add to the reduction for this thread.
/// \param[in] shared A pointer to the start of a shared memory block to use.
///                   This should be at least the size of the warp.
/// \tparam    T      The type of the value.
template <typename T>
fluidity_device_only T block_broadcast_sum(T val, T* shared) {
  const auto flat_idx = static_cast<int>(flattened_thread_id());
  const auto lane     = flat_idx % warpSize;
  const auto wid      = flat_idx / warpSize;
  const auto warps    = static_cast<std::size_t>(
    std::ceil(
      static_cast<float>(block_size()) / 
      static_cast<float>(warpSize)
    )
  );
        
  // Compute the warp sum, sending the result to all threads in the warp.
  val = warp_broadcast_sum(val);

  // Load the value into shared memory. Here, we need to consider the case that
  // there are less warps than the size of the shared memory, hence the else
  // branch. This is easier and faster than passing the shared memory size to
  // the kernel.
  if (lane == 0) {
    shared[wid] = val;
  } else if (flat_idx >= warps && flat_idx < warpSize) {
    shared[flat_idx] = T{0};
  }
  __syncthreads();

  // Reload the results back from the shared memory, and then perform another
  // broadcast sum of all warp sums to send the result to all threads.
  val = shared[lane];

  return warp_broadcast_sum(val);
}

/// Solves the Eikonal equation with a constant speed function of $f = 1$,
/// using the \p input data as the input data, and writing the results to the
/// \p output data. The \p solver is the solver implementation which is used.
///
/// This overload is enabled when the \p input and \p output data are gpu
/// iterators.
///
/// \param[in] in_it             An iterator to the input data.
/// \param[in] out_it            An iterator to the output data.
/// \param[in] dh                The resolution of the solution space.
/// \param[in] conv_it           Mask for if the block has converged.
/// \tparam    Iterator          The type of the input and output iterators.
/// \tparam    T                 The type of the resolution.
/// \tparam    ConvergedIterator The type of the convergence iterator.
template <typename Iterator, typename T, typename ConvergedIterator>
fluidity_global auto fast_iterative_solve(
  Iterator          in_it  ,
  Iterator          out_it ,
  T                 dh     ,
  ConvergedIterator conv_it
) ->void  {
  using iter_t   = std::decay_t<Iterator>;
  using value_t  = typename iter_t::value_t;
  using solver_t = scheme::GodunovUpwindScheme;

  const auto solver        = solver_t{};
  constexpr auto f         = value_t{cx::one};
  constexpr auto max_v     = std::numeric_limits<value_t>::max();
  constexpr auto root_2_d2 = static_cast<value_t>(cx::root_2) / value_t{2};
  constexpr auto tolerance = value_t{1e-8};
  constexpr auto padding   = 1;

  // Shared memory data for if a thread must try again.
  __shared__ int try_again_data[warpSize];

  // Need shared memory iterators for the output
  // data and if a cell is in the active list.
  auto shared_out_it = make_multidim_iterator<iter_t, 1>(out_it);
  auto in_list       = make_multidim_iterator<bool>(Num<iter_t::dimensions>());

  // Offset the iterators ...
  unrolled_for<iter_t::dimensions>([&] (auto dim) {
    in_list.shift(thread_id(dim), dim);
    shared_out_it.shift(thread_id(dim) + padding, dim);
    out_it.shift(flattened_id(dim), dim);
    conv_it.shift(block_id(dim) + 1, dim);
  });

  // Load in the padding. This branching is not good, considering the amount
  // of time this needs to be done, this really needs to be improved.
  // TODO: Optimize this ...
  auto block_active_threads = std::size_t{1};
  unrolled_for<iter_t::dimensions>([&] (auto dim) {
    if (flattened_id(dim) == 0) { 
      *shared_out_it.offset(-1, dim) = max_v;
    } else if (flattened_id(dim) >= (out_it.size(dim) - 1)) {
      *shared_out_it.offset(1, dim) = max_v;
    } else if (thread_id(dim) == 0) {
      *shared_out_it.offset(-1, dim) = *out_it.offset(-1, dim);
    } else if (thread_id(dim) == (block_size(dim) - 1)) {
      *shared_out_it.offset(1, dim) = *out_it.offset(1, dim);
    }
  });

  // Load in the main part of the shared data:
  *shared_out_it = *out_it;
  *in_list       = false;
  __syncthreads();

  auto p = *shared_out_it, q = solver.solve(shared_out_it, dh, f);

  // Here we initialize the source nodes ...
  if (*shared_out_it > q && in_range(out_it)) {
    *in_list = true;
  }
  //__syncthreads();

  bool started_in_list = false; // If the cell was initially in the list.
  // If the cell needs to try and solve again. We need an int here for the
  // special case that no cells are in the list for the block, and then the
  // first reduction will match the block size, and we can exit early.
  int try_again = block_broadcast_sum(
    *shared_out_it == max_v && in_range(out_it)
      ? int{1}
      : static_cast<int>(*in_list), &try_again_data[0]
  ); 

  // Early exit, block is done ...
  if (!try_again) {
    if (flattened_thread_id() == 0) {
      *conv_it = true;
    }
    return;
  } 

  // Limit iterations, if the block is not completely solved then it will be
  // updated in the next iteration.
  auto iters               = 0;
  constexpr auto max_iters = 30;
  while (try_again && ++iters < max_iters) {
    if (in_range(out_it)) {
      started_in_list = *in_list;

      // Part A: For all cells that are in the list, compute the new solution. If
      // the cell has converged then neighbour cells need to update themselves and
      // potentially add themselves to the list. We set in_list = true so that
      // neighbours can add themselves in the next step. We then also need to
      // remove converged cells from the list **but only after neighbours have
      // seen that the cell has converged**.
      if (started_in_list) {
        p              = *shared_out_it;
        q              = solver.solve(shared_out_it, dh, f);
        *shared_out_it = q;
        *in_list       = (std::abs(p - q) < tolerance) ? false : true;
      }


      // Patt B: Check if any of the neighbours have converged and are in the
      // list. If they are, the cell must try and add itself to the list.
      if (!started_in_list) {
        q = solver.solve(shared_out_it, dh, f);
        if (*shared_out_it > q) {
          *shared_out_it = q;
          *in_list       = true;
        }
      }
    }
  
    try_again = block_broadcast_sum(
      *shared_out_it == max_v 
        ? int{1}
        : static_cast<int>(*in_list), &try_again_data[0]
    );
  }
  
  if (flattened_thread_id() == 0) {
    *conv_it = !try_again;

  // Dont think this is needed ... but need to test!
/*
    unrolled_for<iter_t::dimensions>([&] (auto dim) {
      unrolled_for<2>([&] (auto i) {
        constexpr auto off = (i == 0 ? int{-1} : int{1});
        *conv_it.offset(off, dim) = false;
      });
    });
*/
  }
  *out_it = *shared_out_it;
}

}  // namespace kernel

/// Solves the Eikonal equation with a constant speed function of $f = 1$,
/// using the \p in_it iterator data as the input data, and writing the results
/// to the \p out_it output data. 
///
/// TODO: Add solver ...
///
/// \param[in] in_it    An iterator to the input data.
/// \param[in] out_it   An iterator to the output data.
/// \param[in] dh       The resolution of the solution space.
/// \tparam    Iterator The type of the input and output iterators.
/// \tparam    T        The type of the resolution.
template <typename Iterator, typename T>
void fast_iterative(Iterator&& in_it, Iterator&& out_it, T dh) {
  using block_conv_t = bool;
  using iter_t       = std::decay_t<Iterator>;

  auto threads = get_thread_sizes(in_it);
  auto blocks  = get_block_sizes(in_it, threads);

  kernel::fast_iterative_init<<<blocks, threads>>>(in_it, out_it, dh);

  // Create an array of bools which specify 
  // if each of the nodes have converged or not.
  auto converged         = DeviceTensor<block_conv_t, iter_t::dimensions>();
  auto active_blocks     = std::size_t{1};
  auto it_bound          = 0;
  constexpr auto padding = 1;

  unrolled_for<iter_t::dimensions>([&] (auto d) {
    auto dim_size = (d == 0 ? blocks.x : (d == 1 ? blocks.y : blocks.z));

    active_blocks *= dim_size;
    it_bound      += dim_size * dim_size;

    // Resize the dim, adding for padding.
    converged.resize_dim(d, dim_size + (padding << 1));
  });
  fill(converged.multi_iterator(), [&] fluidity_host_device (auto& conv) {
    *conv = false;
  });
  it_bound = std::sqrt(it_bound) * 20;
  
  //const auto total_blocks = converged.total_size() - 2 * iter_t::dimensions;
  
  // Computes the number of blocks which have converged, where conv_mask is the
  // mask which specifies if a block has converged, and iteration is the
  // iteration for which the number of converged blocks for.
  auto num_blocks_converged = [&] (auto&& conv_mask, auto iteration) {
    int converged_blocks = 0;
    for (auto has_block_converged : conv_mask) {
      converged_blocks += has_block_converged;
    }
    //==--- [DEBUG] --------------------------------------------------------==//
    printf(
      "Iters : %3i -- "
      "Blocks Converged: %3i --"
      "Total Active Blocks : %3lu\n",
      iteration                     ,
      converged_blocks              ,
      active_blocks
    );
    return converged_blocks;
  };

  int iters          = 0;
  bool all_converged = false;
  while (!all_converged && iters++ < it_bound) {
    kernel::fast_iterative_solve<<<blocks, threads>>>(
      in_it, out_it, dh, converged.multi_iterator()
    );
    fluidity_check_cuda_result(cudaDeviceSynchronize());
  
    all_converged =
      num_blocks_converged(converged.as_host(), iters) >= active_blocks;
  }

  // Fix sign ... 
  kernel::set_signs<<<blocks, threads>>>(in_it, out_it, dh);
  fluidity_check_cuda_result(cudaDeviceSynchronize());

  // Compute the gradient error sue to the sign change ...
  kernel::compute_gradient_error<<<blocks, threads>>>(out_it, dh);
  fluidity_check_cuda_result(cudaDeviceSynchronize());

  // Fix the gradient errors ..
  kernel::fix_gradient_error<<<blocks, threads>>>(out_it, dh);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
}

}}} // namespace fluid::solver::cuda

#endif // FLUIDITY_SCHEME_CUDA_FAST_ITERATIVE_METHOD_CUH
