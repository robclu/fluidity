//==--- fluidity/reconstruction/basic_reconstructor.hpp- -----*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  basic_reconstructor.hpp
/// \brief This file defines an implementation of a reconstructor which does
///        nothing to the input data, and simply returns the input states at
///        the forward or backward face for a cell.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_RECONSTRUCTION_BASIC_RECONSTRUCTOR_HPP
#define FLUIDITY_RECONSTRUCTION_BASIC_RECONSTRUCTOR_HPP

#include "reconstructor.hpp"

namespace fluid {
namespace recon {

/// The BasicReconstrctor struct implements the reconstruction interface to
/// allow the reconstruction of state data to get the left and right
/// reconstructed data at the forward or backward face.
/// \tparam Limter The type of the sloper limiter to use.
template <typename Limiter>
struct BasicReconstructor : public Reconstructor<BasicReconstructor<Limiter>> {
 private:
  /// Defines the value of the reconstruction application to a forward face.
  static constexpr int  forward_face  = 1;
  /// Defiens the value of the reconstruction application to a backward face.
  static constexpr int  backward_face = -1;

 public:
  /// Defines the type of the limiter.
  using limiter_t = Limiter;

  /// Sets the number of elements which are required in the backward and forward
  /// directions during the reconstruction process.
  static constexpr size_t width = 1;

  /// Constructor, required so that the reconstructor can be created on both the
  /// host and the device.
  fluidity_host_device constexpr BasicReconstructor() {}
  
  /// Returns the left input state in the forward direction, where the forward
  /// direction is one of:
  ///
  ///   { right (x-dim), up (y-dim), inward (z-dim) }
  /// 
  /// \param[in]  state_it   The state iterator to reconstruct from.
  /// \tparam     Iterator   The type of the iterator, which must be
  ///                        multi-dimensional 
  /// \tparam     Material   The type of the material.
  /// \tparam     T          The type of the scaling factor.
  /// \tparam     Value      The value which defines the dimension.
  template <typename Iterator, typename Material, typename T, std::size_t Value>
  fluidity_host_device constexpr auto
  input_fwrd_left(Iterator&& state_it, Material&&, T, Dimension<Value>) const
  {
    return *state_it;
  }

  /// Returns the left right state in the forward direction, where the forward
  /// direction is one of:
  ///
  ///   { right (x-dim), up (y-dim), inward (z-dim) }
  /// 
  /// \param[in]  state_it   The state iterator to reconstruct from.
  /// \tparam     Iterator   The type of the iterator, which must be
  ///                        multi-dimensional 
  /// \tparam     Material   The type of the material.
  /// \tparam     T          The type of the scaling factor.
  /// \tparam     Value      The value which defines the dimension.
  template <typename Iterator, typename Material, typename T, std::size_t Value>
  fluidity_host_device constexpr auto
  input_fwrd_right(Iterator&& state_it, Material&&, T, Dimension<Value>) const
  {
    return *state_it.offset(forward_face, Dimension<Value>{});
  }

  /// Returns the left input state in the backward direction, where the backward
  /// direction is one of:
  ///
  ///   { left (x-dim), down (y-dim), outward (z-dim) }
  /// 
  /// \param[in]  state_it   The state iterator to reconstruct from.
  /// \tparam     Iterator   The type of the iterator, which must be
  ///                        multi-dimensional 
  /// \tparam     Material   The type of the material.
  /// \tparam     T          The type of the scaling factor.
  /// \tparam     Value      The value which defines the dimension.
  template <typename Iterator, typename Material, typename T, std::size_t Value>
  fluidity_host_device constexpr auto
  input_back_left(Iterator&& state_it, Material&&, T, Dimension<Value>) const
  {
    return *state_it.offset(backward_face, Dimension<Value>{});
  }

  /// Returns the right state in the backward direction, where the backward
  /// direction is one of:
  ///
  ///   { left (x-dim), down (y-dim), outward (z-dim) }
  /// 
  /// \param[in]  state_it   The state iterator to reconstruct from.
  /// \tparam     Iterator   The type of the iterator, which must be
  ///                        multi-dimensional 
  /// \tparam     Material   The type of the material.
  /// \tparam     T          The type of the scaling factor.
  /// \tparam     Value      The value which defines the dimension.
  template <typename Iterator, typename Material, typename T, std::size_t Value>
  fluidity_host_device constexpr auto
  input_back_right(Iterator&& state_it, Material&&, T, Dimension<Value>) const
  {
    return *state_it;
  }
};

}} // namespace fluid::recon

#endif // FLUIDITY_RECONSTRUCTION_BASIC_RECONSTRUCTOR_HPP