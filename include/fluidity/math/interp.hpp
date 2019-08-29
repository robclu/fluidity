//==--- fluidity/math/interp.hpp --------------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  math.hpp
/// \brief This file defines math functionality.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_MATH_INTERP_HPP
#define FLUIDITY_MATH_INTERP_HPP

#include "math.hpp"
#include <fluidity/container/array.hpp>
#include <fluidity/traits/iterator_traits.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid  {
namespace math   {

/// Interpolates the data around the \p it iterator data using the weights
/// defined by the \p weights. Any weight for a dimension which is greater than
/// 1 is interpreted to mean shift the iterator over in the given dimension. For
/// example, for the iterator pointing to cell A, where the data is as follows:
///
/// \code{.bash}
///   -------------------------
///   |  W  |  X  |  Y  |  Z  |
///   --------------------------
/// \endcode
///
/// as 
///   - $ w = {1.4}  : r = Y * (1.0 - 0.4) + Z * 0.4 $
///   - $ w = {0.2}  : r = X * (1.0 - 0.2) + Y * 0.2 $
///   - $ w = {-0.7} : r = X * {1.0 - 0.7} + W * 0.7 $
///
/// where w are the weights. This overload is enables when the iterator is one
/// dimensional.
///
/// This will fail at compile time if the number of dimensions in the weight
/// container is different from the number of dimensions in the iterator.
///
/// \pre This assumes that the \p it iterator is offset to the appropriate cell
///      from which the interpolation will be performed.
///
/// \param[in] it       The iterator to use to interpolate data for.
/// \param[in] weights  The weights for the nodes in the interpolation.
/// \tparam    Iterator The type of the iterator.
/// \tparam    Weights  The type of the weights.
template <
  typename Iterator,
  typename Weights ,
  traits::it_1d_enable_t<Iterator> = 0
>
fluidity_host_device auto interp(Iterator&& it, Weights&& weights)
-> std::decay_t<decltype(*it)> {
  using ret_t         = std::decay_t<decltype(*it)>;
  using value_t       = std::decay_t<decltype(weights[0])>;
  constexpr auto dims = std::decay_t<Iterator>::dimensions;
  static_assert(
    dims == std::decay_t<Weights>::elements,
    "Number of weights does not match the number of dimensions for the iterator"
  );
      
  // Comptute values used for the interpolation:  
  const auto sign    = math::signum(weights[dim_x]);
  const auto abs_w   = std::abs(weights[dim_x]);
  const auto abs_off = std::floor(abs_w);
  const auto off     = sign * abs_off;
  const auto factor  = abs_w - abs_off;

  return ret_t{
    (*it.offset(off, dim_x)) * (value_t(1) - factor) +
    (*it.offset(off + sign, dim_x)) * factor
  };
}

/// Interpolates the data around the \p it iterator using the weights
/// defined by the \p weights. Any weight for a dimension which is greater than
/// 1 is interpreted to mean shift the iterator over in the given dimension. For
/// example, for the iterator pointing to cell X, using offsets of {1.4, -1.7}
/// will interpolate the following cell centerd data:
///
/// \code{.bash}
///   -------------------------
///   |  X  |     |     |     |
///   -------------------------
///   |     |  A  |  B  |     |
///   -------------------------
///   |     |  C  |  D  |     |
///   -------------------------
/// \endcode
///
/// as:
/// \begin{equation}
///   x1 = A * (1.0 - 0.4)  + B * 0.4
///   x2 = C * (1.0 - 0.4)  + D 8 0.4
///   r  = x1 * (1.0 - 0.7) + x2 * 0.7
/// \end{equation}
///
/// This overload is enables when the iterator is two dimensional.
///
/// This will fail at compile time if the number of dimensions in the weight
/// container is different from the number of dimensions in the iterator.
///
/// \pre This assumes that the \p it iterator is offset to the appropriate cell
///      from which the interpolation will be performed.
///
/// \param[in] it       The iterator to use to interpolate data for.
/// \param[in] weights  The weights for the nodes in the interpolation.
/// \tparam    Iterator The type of the iterator.
/// \tparam    Weights  The type of the weights.
template <
  typename Iterator,
  typename Weights ,
  traits::it_2d_enable_t<Iterator> = 0
>
fluidity_host_device auto interp(Iterator&& it, Weights&& weights)
-> std::decay_t<decltype(*it)> {
  using ret_t         = std::decay_t<decltype(*it)>;
  using value_t       = std::decay_t<decltype(weights[0])>;
  constexpr auto dims = std::decay_t<Iterator>::dimensions;
  static_assert(
    dims == std::decay_t<Weights>::elements,
    "Number of weights does not match the number of dimensions for the iterator"
  );

  // Defines the type of container for the nodes. We need two nodes for each of
  // the dimensions.
  using node_container_t = Array<Array<decltype(*it)&, 2>, 2>;
  auto nodes       = node_container_t(&(*it));
  auto new_weights = weights;

  unrolled_for<2>([&] (auto dim) {
    const auto sign    = math::signum(weights[dim]);
    const auto abs_w   = std::abs(weights[dim]);
    const auto abs_off = std::floor(abs_w);
    const auto off     = sign * abs_off;
    new_weights[dim]   = abs_w - abs_off;

    nodes[dim][0] = *it.offset(off, dim);
    nodes[dim][1] = *it.offset(off + sign, dim);
  });

  // First lerp in first dimension.
  const auto factor = value_t(1) - new_weights[dim_x];
  const auto close  = nodes[dim_x][0] * factor
                    + nodes[dim_x][1] * new_weights[dim_x];
  const auto far    = nodes[dim_y][0] * factor
                    + nodes[dim_y][1] * new_weights[dim_x];

  // Next lerp the results in the next dimension.
  return ret_t{
    close * (value_t{1} - new_weights[dim_y]) +
    far * new_weights[dim_y]
  };
}

/// Interpolates the data around the \p it iterator using the weights
/// defined by the \p weights. Any weight for a dimension which is greater than
/// 1 is interpreted to mean shift the iterator over in the given dimension.
///
/// This performs a 2D bi-linear interpolation in each of the relevant planes
/// in the z-dimension, where the location of the planes is defined by the
/// weight for the z-dimension and the current location of the \p it iterator.
/// 
/// The results of the 2D bi-linear interpolation are then used with the
/// z-dimension weight in a linear interpolation to compute the final result.
///
/// This overload is enables when the iterator is three dimensional.
///
/// This will fail at compile time if the number of dimensions in the weight
/// container is different from the number of dimensions in the iterator.
///
/// \pre This assumes that the \p it iterator is offset to the appropriate cell
///      from which the interpolation will be performed.
///
/// \param[in] it       The iterator to use to interpolate data for.
/// \param[in] weights  The weights for the nodes in the interpolation.
/// \tparam    Iterator The type of the iterator.
/// \tparam    Weights  The type of the weights.
template <
  typename Iterator,
  typename Weights ,
  traits::it_3d_enable_t<Iterator> = 0
>
fluidity_host_device auto interp(Iterator&& it, Weights&& weights)
-> std::decay_t<decltype(*it)> {
  using ret_t         = std::decay_t<decltype(*it)>;
  using value_t       = std::decay_t<decltype(weights[0])>;
  constexpr auto dims = std::decay_t<Iterator>::dimensions;
  static_assert(
    dims == std::decay_t<Weights>::elements,
    "Number of weights does not match the number of dimensions for the iterator"
  );

  // Defines the type of container for the nodes. We need 2 x (2 x 2).
  using node_container_t = Array<Array<Array<decltype(*it)&, 2>, 2>, 2>;
  auto nodes       = node_container_t(&(*it));
  auto new_weights = weights;

  // First in the z-dimension, offset the nodes:
  auto sign_z    = math::signum(weights[dim_z]);
  auto abs_w_z   = std::abs(weights[dim_z]);
  auto abs_off_z = std::floor(abs_w_z);
  auto off_z     = sign_z * abs_off_z;

  for (auto i : range(2)) {
    auto it_z = it.offset(sign_z * abs_off_z + i * sign_z, dim_z);

    unrolled_for<2>([&] (auto dim) {
      const auto sign    = math::signum(weights[dim]);
      const auto abs_w   = std::abs(weights[dim]);
      const auto abs_off = std::floor(abs_w);
      const auto off     = sign * abs_off;
      new_weights[dim]   = abs_w - abs_off;

      nodes[i][dim][0] = *it_z.offset(off, dim);
      nodes[i][dim][1] = *it_z.offset(off + sign, dim);
    });
  }

  // First lerp in first dimension
  auto final_nodes = Array<std::decay_t<decltype(*it)>, 2>();
  for (auto i : range(2)) {
    const auto factor = value_t(1) - new_weights[dim_x];
    const auto close_temp = nodes[i][dim_x][0] * factor
                          + nodes[i][dim_x][1] * new_weights[dim_x];
    const auto far_temp   = nodes[i][dim_y][0] * factor
                          + nodes[i][dim_y][1] * new_weights[dim_x];

    final_nodes[i] = close_temp * (value_t{1} - new_weights[dim_y])
                   + far_temp * new_weights[dim_y];
  }

  // Final lerp:
  return ret_t{ 
    final_nodes[0] * (value_t(1) - new_weights[dim_z]) +
    final_nodes[1] * new_weights[dim_z]
  };
}

}} // namespace fluid::math

#endif // FLUIDITY_MATH_INTERP_HPP
