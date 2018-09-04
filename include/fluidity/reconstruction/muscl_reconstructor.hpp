//==--- fluidity/reconstruction/muscl_reconstructor.hpp- -----*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  muscl_reconstructor.hpp
/// \brief This file defines an implementation of the MUSCL-Hanconk
///        reconstruction technique.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_RECONSTRUCTION_MUSCL_RECONSTRUCTOR_HPP
#define FLUIDITY_RECONSTRUCTION_MUSCL_RECONSTRUCTOR_HPP

#include "reconstructor.hpp"

#include <fluidity/dimension/thread_index.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid {
namespace recon {

/// The MHReconstructor struct implements the reconstruction interface to allow
/// the reconstruction of state data using the MUSCL-Hancock method.
/// \tparam Limter The type of the sloper limiter to use.
template <typename Limiter>
struct MHReconstructor : public Reconstructor<MHReconstructor<Limiter>> {
 private:
  /// Defines the type of the slope limiter used by the reconstructor.
  using limiter_t = Limiter;

  /// Defines the value of the reconstruction application to a right face.
  static constexpr auto right_face = int{1};
  /// Defiens the value of the reconstruction application to a left face.
  static constexpr auto left_face  = int{-1};

  /// Defines a struct which can be used to overload implementations of the
  /// reconstruction for different faces.
  /// \tparam Face The face to overload for.
  template <int Face> struct FaceSelector {};

  /// Defines an alias for a right face overload.
  using right_face_t = FaceSelector<right_face>;
  /// Defines an alias for a left face overload.
  using left_face_t  = FaceSelector<left_face>;

  /// Defines an intance of a left face overload type.
  static constexpr auto left_face_select  = left_face_t{};
  /// Defines an intance of a right face overload type.
  static constexpr auto right_face_select = right_face_t{};

  /// Creates the evolution of the data, returning the appropriate reconstructed
  /// data. This overload returns the evolved data for the right face in the
  /// x-direction (up face in the y dimension, in face in the z direction).
  /// \param[in] state    An iterator over the state data.
  /// \param[in] mat      The material in which the faces lie.
  /// \param[in] dtdh     The space time scaling factor.
  /// \tparam    IT       The type of the state iterator.
  /// \tparam    Mat      The type of the material.
  /// \tparam    T        The type of the scaling factor.
  /// \tparam    Dim      The type of the dimension.
  template <typename IT, typename Mat, typename T, typename Dim> 
  fluidity_host_device static auto
  evolve(IT&& state, Mat&& mat, T dtdh, Dim dim, right_face_t) 
  {
    using state_t = std::decay_t<decltype(*state)>;
    using value_t = std::decay_t<T>;

    constexpr auto half    = value_t{0.5};
    const auto     limiter = limiter_t();
    const auto     delta   = half * limiter.limit(state, mat, dim);

    // Boundary extrapolated value:
    // U_i^n \pm \frac{1}{2} \delta \eita i
    const auto bev = state_t{*state + delta};

    // Evolve BEV in time:
    return state_t{
      bev + 
      (half * dtdh) * 
      (state_t{*state - delta}.flux(mat, dim) - bev.flux(mat, dim))
    };
  }

  /// Creates the evolution of the data, returning the appropriate reconstructed
  /// data. This overload returns the evolved data for the left face in the
  /// x-direction (down face in the y dimension, out face in the z direction).
  /// \param[in] state    An iterator over the state data.
  /// \param[in] mat      The material in which the faces lie.
  /// \param[in] dtdh     The space time scaling factor.
  /// \tparam    IT       The type of the state iterator.
  /// \tparam    Mat      The type of the material.
  /// \tparam    T        The type of the scaling factor.
  /// \tparam    Dim      The type of the dimension.
  template <typename IT, typename Mat, typename T, typename Dim> 
  fluidity_host_device static auto
  evolve(IT&& state, Mat&& mat, T dtdh, Dim dim, left_face_t)
  {
    using state_t = std::decay_t<decltype(*state)>;
    using value_t = std::decay_t<T>;

    constexpr auto half    = value_t{0.5};
    const auto     limiter = limiter_t();
    const auto     delta   = half * limiter.limit(state, mat, dim);

    // Boundary extrapolated value:
    const auto bev = state_t{*state - delta};
    
    // Evolve BEV in time:
    return state_t{
      bev +
      (half * dtdh) *
      (bev.flux(mat, dim) - state_t{*state + delta}.flux(mat, dim))
    };
  }

 public:
  /// Sets the number of elements which are required in the backward and forward
  /// directions during the reconstruction process.
  static constexpr size_t width = limiter_t::width;

  /// Constructor, required so that the reconstructor can be created on both the
  /// host and the device.
  //fluidity_host_device constexpr MHReconstructor() {}

  /// Returns the left input state in the forward direction, where the forward
  /// direction is one of:
  ///
  ///   { right (x-dim), up (y-dim), inward (z-dim) }
  /// 
  /// \param[in]  state   The state iterator to reconstruct from.
  /// \param[in]  mat     The material for the system.
  /// \param[in]  dtdh    The space time scaling factor.
  /// \param[in]  dim     The dimesion for reconstruction.
  /// \tparam     IT      The type of the iterator.
  /// \tparam     Mat     The type of the material.
  /// \tparam     T       The type of the scaling factor.
  /// \tparam     Dim     The type of the dimension.
  template <typename IT, typename Mat, typename T, typename Dim>
  fluidity_host_device constexpr auto
  input_fwrd_left(IT&& state, Mat&& mat, T dtdh, Dim dim) const
  {
    return evolve(state, mat, dtdh, dim, right_face_select);
  }

  /// Returns the left right state in the forward direction, where the forward
  /// direction is one of:
  ///
  ///   { right (x-dim), up (y-dim), inward (z-dim) }
  /// 
  /// \param[in]  state   The state iterator to reconstruct from.
  /// \param[in]  mat     The material for the system.
  /// \param[in]  dtdh    The space time scaling factor.
  /// \param[in]  dim     The dimesion for reconstruction.
  /// \tparam     IT      The type of the iterator.
  /// \tparam     Mat     The type of the material.
  /// \tparam     T       The type of the scaling factor.
  /// \tparam     Dim     The type of the dimension.
  template <typename IT, typename Mat, typename T, typename Dim>
  fluidity_host_device constexpr auto
  input_fwrd_right(IT&& state, Mat&& mat, T dtdh, Dim dim) const
  {
    return
      evolve(state.offset(right_face, dim), mat, dtdh, dim, left_face_select);
  }

  /// Returns the left input state in the backward direction, where the backward
  /// direction is one of:
  ///
  ///   { left (x-dim), down (y-dim), outward (z-dim) }
  /// 
  /// \param[in]  state   The state iterator to reconstruct from.
  /// \param[in]  mat     The material for the system.
  /// \param[in]  dtdh    The space time scaling factor.
  /// \param[in]  dim     The dimesion for reconstruction.
  /// \tparam     IT      The type of the iterator.
  /// \tparam     Mat     The type of the material.
  /// \tparam     T       The type of the scaling factor.
  /// \tparam     Dim     The type of the dimension.
  template <typename IT, typename Mat, typename T, typename Dim>
  fluidity_host_device constexpr auto
  input_back_left(IT&& state, Mat&& mat, T dtdh, Dim dim) const
  {
    return 
      evolve(state.offset(left_face, dim), mat, dtdh, dim, right_face_select);
  }

  /// Returns the right state in the backward direction, where the backward
  /// direction is one of:
  ///
  ///   { left (x-dim), down (y-dim), outward (z-dim) }
  /// 
  /// \param[in]  state   The state iterator to reconstruct from.
  /// \param[in]  mat     The material for the system.
  /// \param[in]  dtdh    The space time scaling factor.
  /// \param[in]  dim     The dimesion for reconstruction.
  /// \tparam     IT      The type of the iterator.
  /// \tparam     Mat     The type of the material.
  /// \tparam     T       The type of the scaling factor.
  /// \tparam     Dim     The type of the dimension.
  template <typename IT, typename Mat, typename T, typename Dim>
  fluidity_host_device constexpr auto
  input_back_right(IT&& state, Mat&& mat, T dtdh, Dim dim) const
  {
    return evolve(state, mat, dtdh, dim, left_face_select);
  }
};

}} // namespace fluid::recon

#endif // FLUIDITY_RECONSTRUCTION_MUSCL_RECONSTRUCTOR_HPP