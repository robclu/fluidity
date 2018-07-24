//==--- fluidity/dimension/dimension.hpp ------------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  dimension.hpp
/// \brief This file defines dimension related concepts.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_DIMENSION_DIMENSION_HPP
#define FLUIDITY_DIMENSION_DIMENSION_HPP

#include <fluidity/utility/portability.hpp>
#include <cstddef>

namespace fluid {

/// Defines a class to represent a dimension, where the value is known at
/// compile time. The class should be used through the aliases when a single
/// dimension must be specified, i.e:
/// 
/// ~~~cpp
/// // Not clear, what is 0?
/// do_something(container, 0);
/// 
/// // More clear, intention of x dimension application is known at call site.
/// do_something(container, dim_x);
/// ~~~
/// 
/// The other use case is when used with `unrolled_for`, where is can be used
/// more generically:
/// 
/// ~~~cpp
/// unrolled_for<num_dimensions>([] (auto dim_value) {
///   // Compile time dimension can be used here:
///   do_something(container, Dimension<dim_value>{});
/// })
/// ~~~
/// 
/// \tparam Value   The value of the dimension.
template <std::size_t Value>
struct Dimension {
  /// Returns the value of the dimension.
  static constexpr std::size_t value = Value;

  /// Overload of operator size_t to convert a dimension to a size_t.
  constexpr operator size_t() const
  {
    return static_cast<std::size_t>(Value);
  }
};

// If the compilation system has cuda functionality then set the default
// execution policy to use the GPU.
#if defined(FLUIDITY_CUDA_AVAILABLE)

/// Defines the type for runtime dimension info.
using dim3_t = dim3;

#else

/// Defines a class which mimics cuda's dim3 type.
struct Dim3 {
  int x = 0;  //!< Value in the x dimension.  
  int y = 0;  //!< Value in the y dimension.
  int z = 0;  //!< Value in the z dimension.
};

/// Defines the type for runtime dimension information.
using dim3_t = Dim3;

#endif // FLUIDITY_CUDA_AVAILABLE

/// Aliases for the x spacial dimension type.
using dimx_t = Dimension<0>;
/// Alias for the y spacial dimension type.
using dimy_t = Dimension<1>;
/// Aloas for the z spacial dimension type.
using dimz_t = Dimension<2>;

/// Defines a compile time type for the x spacial dimension.
static constexpr dimx_t dim_x = dimx_t{};
/// Defines a compile time type for the x spacial dimension.
static constexpr dimy_t dim_y = dimy_t{};
/// Defines a compile time type for the x spacial dimension.
static constexpr dimz_t dim_z = dimz_t{};

#if !defined(FLUIDITY_DEFAULT_THREADS_1D_X)
  /// Defines the default number of threads per block.
  static constexpr std::size_t threads_per_block_1d_x = 512;
#else
  /// Defines the default number of threads per block.
  static constexpr std::size_t threads_per_block_1d_x = 
    FLUIDITY_DEFAULT_THREADS_1D_X
#endif

#if !defined(FLUIDITY_DEFAULT_THREADS_2D_X)
  /// Defines the default number of threads per block.
  static constexpr std::size_t threads_per_block_2d_x = 32;
#else
  /// Defines the default number of threads per block.
  static constexpr std::size_t threads_per_block_2d_x =
    FLUIDITY_DEFAULT_THREADS_2D_X;
#endif

#if !defined(FLUIDITY_DEFAULT_THREADS_2D_Y)
  /// Defines the default number of threads per block.
  static constexpr std::size_t threads_per_block_2d_y = 16;
#else
  /// Defines the default number of threads per block.
  static constexpr std::size_t threads_per_block_2d_y =
    FLUIDITY_DEFAULT_THREADS_2D_Y;
#endif

#if !defined(FLUIDITY_DEFAULT_THREADS_3D_X)
  /// Defines the default number of threads per block.
  static constexpr std::size_t threads_per_block_3d_x = 8;
#else
  /// Defines the default number of threads per block.
  static constexpr std::size_t threads_per_block_3d_x = 
    FLUIDITY_DEFAULT_THREADS_3D_X;
#endif

#if !defined(FLUIDITY_DEFAULT_THREADS_3D_Y)
  /// Defines the default number of threads per block.
  static constexpr std::size_t threads_per_block_3d_y = 8;
#else
  /// Defines the default number of threads per block.
  static constexpr std::size_t threads_per_block_3d_y = 
    FLUIDITY_DEFAULT_THREADS_3D_Y;
#endif

#if !defined(FLUIDITY_DEFAULT_THREADS_3D_Z)
  /// Defines the default number of threads per block.
  static constexpr std::size_t threads_per_block_3d_z = 8;
#else
  /// Defines the default number of threads per block.
  static constexpr std::size_t threads_per_block_3d_z = 
    FLUIDITY_DEFAULT_THREADS_3D_Z;
#endif
} // namespace fluid

#endif // FLUIDITY_DIMENSION_DIMENSION_HPP