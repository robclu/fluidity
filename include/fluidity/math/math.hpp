//==--- fluidity/math/math.hpp ----------------------------- -*- C++ -*- ---==//
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

#ifndef FLUIDITY_MATH_MATH_HPP
#define FLUIDITY_MATH_MATH_HPP

#include <fluidity/container/array.hpp>
#include <fluidity/utility/portability.hpp>
#include <type_traits>

namespace fluid  {
namespace math   {

//==--- [abs] --------------------------------------------------------------==//

/// Performs an elementwise abs on an array \p a, returning a new array with the
/// results.
/// \param[in] a The array to compute the elementwise abs for.
/// \tparam    T The type of the data for the array.
/// \tparam    S The size of the array.
template <typename T, std::size_t S>
fluidity_host_device constexpr auto abs(const Array<T, S>& a) -> Array<T, S> {
  auto r = Array<T, S>{};
  unrolled_for_bounded<S>([&] (auto i) {
    r[i] = std::abs(a[i]);
  });
  return r;
}

//==--- [clamp] ------------------------------------------------------------==//

/// Clamps the input \p a between \p min_val and \p max_val.
/// \param[in] a The value to clamp.
/// \param[in] min_val The minimum value in the clamp range.
/// \param[in] max_val The maximum value in the clamp range.
/// \tparam    T The type of the data.
template <typename T>
fluidity_host_device constexpr auto clamp(T a, T min_val, T max_val) -> T {
  return std::max(min_val, std::min(a, max_val));
}

/// Performs an elementwise clamp of each of the inputs in the array\p a
/// between \p min_val and \p max_val.
/// \param[in] a       The array to clamp each element for.
/// \param[in] min_val The minimum value in the clamp range.
/// \param[in] max_val The maximum value in the clamp range.
/// \tparam    T The type of the data.
template <typename T, std::size_t S>
fluidity_host_device constexpr auto clamp(
  const Array<T, S>& a      ,
  T                  min_val,
  T                  max_val
) -> Array<T, S> {
  auto r = Array<T, S>{};
  unrolled_for_bounded<S>([&] (auto i) {
    r[i] = clamp(a[i], min_val, max_val);
  });
  return r;
}

//==--- [dot] --------------------------------------------------------------==//

/// Performs a dot (inner) product of the arrays \p a and \p, returning the
/// result.
/// \param[in] a  The first array for the dot product.
/// \param[in] b  The second array for the dot product.
/// \tparam    T  The type of the data.
/// \tparam    S  The number of elements in the arrays.
template <typename T, std::size_t S>
fluidity_host_device constexpr auto dot(
  const Array<T, S>& a,
  const Array<T, S>& b
) -> T {

  constexpr auto unroll_size      = 4;
  constexpr auto unroll_loops     = S / unroll_size;
  constexpr auto last_unroll_size = S - unroll_loops * unroll_size;

  auto sum = T{0};
  for (auto outer_idx : range(unroll_loops)) {
    unrolled_for<unroll_size>([&] (auto inner_idx) {
      const auto i = outer_idx * unroll_size + inner_idx;
      sum += a[i] * b[i];
    });
  }
  
  unrolled_for<last_unroll_size>([&] (auto i) {
    sum += a[i] * b[i];
  });
  return sum;
}

/// Performs a dot (inner) product of the arrays \p a and \p, returning the
/// result.
/// \param[in] a  The first array for the dot product.
/// \param[in] b  The second array for the dot product.
/// \tparam    T  The type of the data in the first array.
/// \tparam    U  The type of the data in the second array.
/// \tparam    S  The number of elements in the arrays.
template <typename T, typename U, std::size_t S>
fluidity_host_device constexpr auto dot(
  const Array<T, S>& a,
  const Array<U, S>& b
) -> T {

  constexpr auto unroll_size      = 4;
  constexpr auto unroll_loops     = S / unroll_size;
  constexpr auto last_unroll_size = S - unroll_loops * unroll_size;

  auto sum = T{0};
  for (auto outer_idx : range(unroll_loops)) {
    unrolled_for<unroll_size>([&] (auto inner_idx) {
      const auto i = outer_idx * unroll_size + inner_idx;
      sum += a[i] * b[i];
    });
  }
  
  unrolled_for<last_unroll_size>([&] (auto i) {
    sum += a[i] * b[i];
  });
  return sum;
}

/// Performs a dot (inner) product of the arrays \p a and \p, returning the
/// result.
/// \param[in] a  The first array for the dot product.
/// \param[in] b  The second array for the dot product.
/// \tparam    T  The type of the data.
/// \tparam    S  The number of elements in the arrays.
template <typename T, std::size_t S>
fluidity_host_device constexpr auto dot(Array<T, S>&& a, Array<T, S>&& b) -> T {
  constexpr auto unroll_size      = 4;
  constexpr auto unroll_loops     = S / unroll_size;
  constexpr auto last_unroll_size = S - unroll_loops * unroll_size;

  auto sum = T{0};
  for (auto outer_idx : range(unroll_loops)) {
    unrolled_for<unroll_size>([&] (auto inner_idx) {
      const auto i = outer_idx * unroll_size + inner_idx;
      sum += a[i] * b[i];
    });
  }
  
  unrolled_for<last_unroll_size>([&] (auto i) {
    sum += a[i] * b[i];
  });
  return sum;
}

/// Performs a dot (inner) product of the arrays \p a and \p, returning the
/// result.
/// \param[in] a  The first array for the dot product.
/// \param[in] b  The second array for the dot product.
/// \tparam    T  The type of the data in the first array.
/// \tparam    U  The type of the data in the second array.
/// \tparam    S  The number of elements in the arrays.
template <typename T, typename U, std::size_t S>
fluidity_host_device constexpr auto dot(Array<T, S>&& a, Array<U, S>&& b) -> T {
  constexpr auto unroll_size      = 4;
  constexpr auto unroll_loops     = S / unroll_size;
  constexpr auto last_unroll_size = S - unroll_loops * unroll_size;

  auto sum = T{0};
  for (auto outer_idx : range(unroll_loops)) {
    unrolled_for<unroll_size>([&] (auto inner_idx) {
      const auto i = outer_idx * unroll_size + inner_idx;
      sum += a[i] * b[i];
    });
  }
  
  unrolled_for<last_unroll_size>([&] (auto i) {
    sum += a[i] * b[i];
  });
  return sum;
}

//==--- [length] -----------------------------------------------------------==//

/// Returns the length (eucledian distance) of an array.
/// \param[in] a The array to compute the length of.
/// \tparam    T The type of the data for the array.
/// \tparam    S The size of the array.
template <typename T, std::size_t S>
fluidity_host_device constexpr auto length(const Array<T, S>& a) -> T {
  auto sum = T{0};
  unrolled_for_bounded<S>([&] (auto i) {
    sum += a[i] * a[i];
  });
  return std::sqrt(sum);
}

//==--- [max] --------------------------------------------------------------==//

/// Returns the max value of \p a and \p b.
/// \param[in] a The value of the first element for the max op.
/// \param[in] b The value of the second element for the max op.
/// \tparam    T The data type for the array and the other value.
template <typename T>
fluidity_host_device constexpr auto max(T a, T b) -> T {
  return std::max(a, b);
}

/// Performs an elementwise max for the two arrays \p a and \p b, returning an
/// array of the same size.
/// \param[in] a The array for the first element for the max op.
/// \param[in] b The array for the second element for the max op.
/// \tparam    T The data type for the array.
/// \tparam    S The number of elements in the array.
template <typename T, std::size_t S>
fluidity_host_device constexpr auto 
max(const Array<T, S>& a, const Array<T, S>& b) -> Array<T, S> {
  auto r = Array<T, S>{};
  unrolled_for_bounded<S>([&] (auto i) {
    r[i] = std::max(a[i], b[i]);
  });
  return r;
}

/// Performs an elementwise max for each element of the array \p a and the value
/// \p b, returning an array of the same size as array \p a.
/// \param[in] a The array for the first element for the max op.
/// \param[in] b The value of the second element for the max op.
/// \tparam    T The data type for the array and the other value.
/// \tparam    S The number of elements in the array.
template <typename T, std::size_t S>
fluidity_host_device constexpr auto max(const Array<T, S>& a, T b)
    -> Array<T, S> {
  auto r = Array<T, S>{};
  unrolled_for_bounded<S>([&] (auto i) {
    r[i] = max(a[i], b);
  });
  return r;
}

//==--- [min] --------------------------------------------------------------==//

/// Performs an elementwise min for each element of the array \p a and the value
/// \p b, returning an array of the same size as array \p a. This is a wrapper
/// around the std lib version.
/// Returns the min value of \p a and \p b.
/// \param[in] a The value of the first element for the min op.
/// \param[in] b The value of the second element for the min op.
/// \tparam    T The data type for the array and the other value.
template <typename T>
fluidity_host_device constexpr auto min(T a, T b) -> T {
  return std::min(a, b);
}

/// Performs an elementwise min for the two arrays \p a and \p b, returning an
/// array of the same size.
/// \param[in] a The array for the first element for the min op.
/// \param[in] b The array for the second element for the min op.
/// \tparam    T The data type for the array.
/// \tparam    S The number of elements in the array.
template <typename T, std::size_t S>
fluidity_host_device constexpr auto min(
  const Array<T, S>& a,
  const Array<T, S>& b
) -> Array<T, S> {
  auto r = Array<T, S>{};
  unrolled_for_bounded<S>([&] (auto i) {
    r[i] = min(a[i], b[i]);
  });
  return r;
}

/// Performs an elementwise min for each element of the array \p a and the value
/// \p b, returning an array of the same size as array \p a.
/// \param[in] a The array for the first element for the min op.
/// \param[in] b The value of the second element for the min op.
/// \tparam    T The data type for the array and the other value.
/// \tparam    S The number of elements in the array.
template <typename T, std::size_t S>
fluidity_host_device constexpr auto min(const Array<T, S>& a, T b)
-> Array<T, S> {
  auto r = Array<T, S>{};
  unrolled_for_bounded<S>([&] (auto i) {
    r[i] = min(a[i], b);
  });
  return r;
}

//==--- [signum] -----------------------------------------------------------==//

namespace detail {

/// Returns -1.0 if x is less than 0.0, 0.0 if x is equal to 0.0, and +1.0 if x
/// is greater than 0. Specialization for an unsigned type.
/// \param[in]  x      The value to get the sign of.
/// \tparam     T      The type of the data.
template <typename T>
fluidity_host_device constexpr auto signum(T x, std::false_type) -> T {
  return T(0) < x;
}

/// Returns -1.0 if x is less than 0.0, 0.0 if x is equal to 0.0, and +1.0 if x
/// is greater than 0. Specialization for an signed type.
/// \param[in]  x      The value to get the sign of.
/// \tparam     T      The type of the data.
template <typename T>
fluidity_host_device constexpr auto signum(T x, std::true_type) -> T {
  return (T(0) < x) - (x < T(0));
}

} // namespace detail

/// Returns -1.0 if x is less than 0.0, 0.0 if x is equal to 0.0, and +1.0 if x
/// is greater than 0. Interface function.
/// \param[in]  x  The value to get the sign of.
/// \tparam     T  The type of the data.
template <typename T>
fluidity_host_device constexpr auto signum(T x) -> T {
  return detail::signum(x, std::is_signed<T>());
}

/// Performs an elementwise signum for the array \p a, returning a new array
/// with the result.
/// \param[in] a The array to perform the signum operation on.
/// \tparam    T The data type for the array.
/// \tparam    S The number of elements in the array.
template <typename T, std::size_t S>
fluidity_host_device constexpr auto signum(const Array<T, S>& a)
-> Array<T, S> {
  auto r = Array<T, S>{};
  unrolled_for_bounded<S>([&] (auto i) {
    r[i] = signum(a[i]);
  });
  return r;
}

}} // namespace fluid::math
 
#endif // FLUIDITY_MATH_MATH_HPP
