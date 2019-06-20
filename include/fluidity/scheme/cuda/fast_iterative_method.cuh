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
/// \param[in] input  The input data iterator.
/// \param[in] output The output data iterator.
/// \param[in] width  The width of the band which defines the source.
/// \tparam    I      The type of the input and output iterators.
template <typename I, typename T = typename std::decay_t<I>::value_t>
fluidity_global void fast_iterative_init(I input, I output, T width) {
  using iter_t = std::decay_t<I>;
  constexpr auto max_v = std::numeric_limits<T>::max();

  // Offset the input and output iterators to their respective places ...
  unrolled_for<iter_t::dimensions>([&] (auto dim) {
    input.shift(flattened_id(dim), dim);
    output.shift(flattened_id(dim), dim);
  });

  *output = (std::abs(*input) < width ? *input : max_v);
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
  const auto flat_id = static_cast<int>(flattened_thread_id());
  const auto lane    = flat_id % warpSize;
  const auto wid     = flat_id / warpSize;
  const auto warps   = block_size() / warpSize;
        
  // Compute the warp sum, sending the result to all threads in the warp.
  val = warp_broadcast_sum(val);

  // Load the value into shared memory. Here, we need to consider the case that
  // there are less warps than the size of the shared memory, hence the else
  // branch. This is easier and faster than passing the shared memory size to
  // the kernel.
  if (lane == 0) {
    shared[wid] = val;
  } else if (flat_id >= warps && flat_id < warpSize) {
    shared[flat_id] = T{0};
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
/// \param[in] input  The input data to use to initialize the solve state.
/// \param[in] output The output data to write the results to.
/// \param[in] solver The solver which computes the Eikonal solution.
/// \tparam    D      The type of the input and output data.
/// \tparam    S      The type of the solver.
template <typename I, typename T, typename C>
fluidity_global void
fast_iterative_solve(I input, I output, T dh, C converged) {
  using iter_t   = std::decay_t<I>;
  using value_t  = typename iter_t::value_t;
  using solver_t = scheme::GodunovUpwindScheme;

  __shared__ int try_again_data[warpSize];

  const auto solver        = solver_t{};
  constexpr auto f         = value_t{cx::one};
  constexpr auto max_v     = std::numeric_limits<value_t>::max();
  constexpr auto root_2_d2 = static_cast<value_t>(cx::root_2) / value_t{2};
  constexpr auto tolerance = value_t{1e-6};

  // Need shared memory iterators for 
  auto out     = make_multidim_iterator<iter_t, 1>(output);
  auto in_list = make_multidim_iterator<bool>(Num<iter_t::dimensions>());

  // Early exit outside threads. This is causing a problem with the broadcast
  // sum ...
  if (!in_range(output)) {
    return;
  }

  // Offset the iterators ...
  unrolled_for<iter_t::dimensions>([&] (auto dim) {
    out.shift(thread_id(dim) + 1, dim);
    in_list.shift(thread_id(dim), dim);
    output.shift(flattened_id(dim), dim);
    converged.shift(block_id(dim) + 1, dim);
  });

  // Load in the padding. This branching is not good, considering the amount
  // of time this needs to be done, this really needs to be improved.
  // TODO: Optimize this ...
  auto block_active_threads = std::size_t{1};
  unrolled_for<iter_t::dimensions>([&] (auto dim) {
    if (flattened_id(dim) == 0) { 
      *out.offset(-1, dim) = max_v;
    } else if (flattened_id(dim) >= output.size(dim) - 1) {
      *out.offset(1, dim) = max_v;
    } else if (thread_id(dim) == 0) {
      *out.offset(-1, dim) = *output.offset(-1, dim);
    } else if (thread_id(dim) == block_size(dim) - 1) {
      *out.offset(1, dim) = *output.offset(1, dim);
    }
  });

  // Load in the main part of the shared data:
  *out     = *output;
  *in_list = false;
  __syncthreads();

  auto p = *out, q = solver.solve(out, dh, f);

  // Here we initialize the source nodes ...
  if (*out > q) {
    *in_list = true;
  }

  bool started_in_list = false; // If the cell was initially in the list.
  // If the cell needs to try and solve again. We need an int here for the
  // special case that no cells are in the list for the block, and then the
  // first reduction will match the block size, and we can exit early.
  int try_again = block_broadcast_sum(
    //static_cast<int>(*in_list), &try_again_data[0]
    *out == max_v ? int{1} : static_cast<int>(*in_list), &try_again_data[0]
  ); 

  // Early exit, block is done ...
  if (!try_again) {
    if (flattened_thread_id() == 0) {
      *converged = true;
    }
    return;
  } 

  // Limit iterations, if the block is not completely solved then it will be
  // updated in the next iteration.
  auto iters               = 0;
  constexpr auto max_iters = 30;
  while (try_again && ++iters < max_iters) {
    started_in_list = *in_list;

    // Part A: For all cells that are in the list, compute the new solution. If
    // the cell has converged then neighbour cells need to update themselves and
    // potentially add themselves to the list. We set in_list = true so that
    // neighbours can add themselves in the next step. We then also need to
    // remove converged cells from the list **but only after neighbours have
    // seen that the cell has converged**.
    if (started_in_list) {
      p        = *out;
      q        = solver.solve(out, dh, f);
      *out     = q;
      *in_list = (std::abs(p - q) < tolerance) ? false : true;
    }


    // Patt B: Check if any of the neighbours have converged and are in the
    // list. If they are, the cell must try and add itself to the list.
    if (!started_in_list) {
      q = solver.solve(out, dh, f);
      if (*out > q) {
        *out = q;
        *in_list = true;
      }
    }
    __syncthreads();
  
    try_again = block_broadcast_sum(
      *out == max_v ? int{1} : static_cast<int>(*in_list), &try_again_data[0]
    );
  }
  
  if (flattened_thread_id() == 0) {
    *converged = !try_again;

/* Dont think this is needed ... but need to test!
    unrolled_for<iter_t::dimensions>([&] (auto dim) {
      unrolled_for<2>([&] (auto i) {
        constexpr auto off = (i == 0 ? int{-1} : int{1});
        *converged.offset(off, dim) = false;
      });
    });
*/
  }
  *output = *out;
}

}  // namespace kernel

/// Solves the Eikonal equation with a constant speed function of $f = 1$,
/// using the \p input data as the input data, and writing the results to the
/// \p output data. The \p solver is the solver implementation which is used.
///
/// This overload is enabled when the \p input and \p output data are gpu
/// iterators.
///
/// \param[in] input  The input data to use to initialize the solve state.
/// \param[in] output The output data to write the results to.
/// \param[in] solver The solver which computes the Eikonal solution.
/// \tparam    D      The type of the input and output data.
/// \tparam    S      The type of the solver.
template <typename I, typename T>
void fast_iterative(I&& input, I&& output, T dh) {
  using block_conv_t = bool;
  using iter_t       = std::decay_t<I>;

  auto threads = get_thread_sizes(input);
  auto blocks  = get_block_sizes(input, threads);

  kernel::fast_iterative_init<<<blocks, threads>>>(input, output, dh);

  // We need an array of bools which specify if each of the nodes have
  // converged or not.
  auto converged     = DeviceTensor<block_conv_t, iter_t::dimensions>();
  auto active_blocks = std::size_t{1};
  auto it_bound      = 0;
  unrolled_for<iter_t::dimensions>([&] (auto d) {
    auto dim_size = (d == 0 ? blocks.x : (d == 1 ? blocks.y : blocks.z));

    active_blocks *= dim_size;
    it_bound      += dim_size * dim_size;

    // Resize the dim, adding for padding.
    converged.resize_dim(d, dim_size + 2);
  });
  fill(converged.multi_iterator(), [&] fluidity_host_device (auto& conv) {
    *conv = false;
  });
  it_bound = std::sqrt(it_bound) * 2;
  
  //const auto total_blocks = converged.total_size() - 2 * iter_t::dimensions;
  auto block_convergence  = [&] (auto&& conv, auto it) {
    int converged_blocks = 0;
    for (auto block_conv : conv) {
      converged_blocks += block_conv;
    }
    printf("Iters : %3i, Blocks Converged: %3i | Total A Blocks : %3lu\n",
      it, converged_blocks, active_blocks);
    return converged_blocks;
  };

  int iters          = 0;
  bool all_converged = false;
  while (!all_converged && iters++ < it_bound) {
    kernel::fast_iterative_solve<<<blocks, threads>>>(
      input, output, dh, converged.multi_iterator()
    );
    fluidity_check_cuda_result(cudaDeviceSynchronize());
  
    all_converged = block_convergence(converged.as_host(), iters) 
                  >= active_blocks;
  }

  // Fix sign ... 
}

}}} // namespace fluid::solver::cuda

#endif // FLUIDITY_SCHEME_CUDA_FAST_ITERATIVE_METHOD_CUH
