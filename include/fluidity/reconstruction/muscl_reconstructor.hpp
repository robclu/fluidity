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

  /// Sets the number of elements which are required in the backward and forward
  /// directions during the reconstruction process.
  static constexpr size_t width = limiter_t::width;

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
  /// \param[in] state_it An iterator over the state data.
  /// \param[in] mat      The material in which the faces lie.
  /// \param[in] dtdh     The space time scaling factor.
  /// \tparam    Iterator The type of the state iterator.
  /// \tparam    Material The type of the material.
  /// \tparam    T        The type of the scaling factor.
  /// \tparam    V        The value which defines the the dimension.
  template <typename Iterator, typename Material, typename T, std::size_t V> 
  fluidity_host_device static auto evolve(Iterator&&       state_it, 
                                          Material&&       mat     ,
                                          T                dtdh    ,
                                          Dimension<V>     /*dim*/ ,
                                          right_face_t     /*face*/)
  {
    using state_t = std::decay_t<decltype(*state_it)>;
    using value_t = std::decay_t<T>;

    constexpr auto half    = value_t{0.5};
    constexpr auto dim     = Dimension<V>{};
    const auto     limiter = limiter_t();
    const auto     delta   = half * limiter.limit(state_it, mat, dim);

    // Boundary extrapolated value:
    // U_i^n \pm \frac{1}{2} \delta \eita i
    const auto bev = state_t{*state_it + delta};

    // Evolve BEV in time:
    return state_t{
      bev + 
      (half * dtdh) * 
      (state_t{*state_it - delta}.flux(mat, dim) - bev.flux(mat, dim))
    };
  }

  /// Creates the evolution of the data, returning the appropriate reconstructed
  /// data. This overload returns the evolved data for the left face in the
  /// x-direction (down face in the y dimension, out face in the z direction).
  /// \param[in] state_it An iterator over the state data.
  /// \param[in] mat      The material in which the faces lie.
  /// \param[in] dtdh     The space time scaling factor.
  /// \tparam    Iterator The type of the state iterator.
  /// \tparam    Material The type of the material.
  /// \tparam    T        The type of the scaling factor.
  /// \tparam    V        The value which defines the the dimension.
  template <typename Iterator, typename Material, typename T, std::size_t V> 
  fluidity_host_device static auto evolve(Iterator&&       state_it, 
                                          Material&&       mat     ,
                                          T                dtdh    ,
                                          Dimension<V>     /*dim*/ ,
                                          left_face_t      /*face*/)
  {
    using state_t = std::decay_t<decltype(*state_it)>;
    using value_t = std::decay_t<T>;

    constexpr auto half    = value_t{0.5};
    constexpr auto dim     = Dimension<V>{};
    const auto     limiter = limiter_t();
    const auto     delta   = half * limiter.limit(state_it, mat, dim);

    // Boundary extrapolated value:
    const auto bev = state_t{*state_it - delta};
    
    // Evolve BEV in time:
    return state_t{
      bev +
      (half * dtdh) *
      (bev.flux(mat, dim) - state_t{*state_it + delta}.flux(mat, dim))
    };
  }

 public:
  /// Constructor, required so that the reconstructor can be created on both the
  /// host and the device.
  fluidity_host_device constexpr MHReconstructor() {}

  /// Returns the left input state in the forward direction, where the forward
  /// direction is one of:
  ///
  ///   { right (x-dim), up (y-dim), inward (z-dim) }
  /// 
  /// \param[in]  state_it   The state iterator to reconstruct from.
  /// \param[in]  material   The material for the system.
  /// \param[in]  dtdh       The space time scaling factor.
  /// \tparam     Iterator   The type of the iterator, which must be
  ///                        multi-dimensional 
  /// \tparam     Material   The type of the material.
  /// \tparam     T          The type of the scaling factor.
  /// \tparam     Value      The value which defines the dimension.
  template <typename Iterator, typename Material, typename T, std::size_t Value>
  fluidity_host_device constexpr auto
  input_fwrd_left(Iterator&&        state_it,
                  Material&&        material, 
                  T                 dtdh    , 
                  Dimension<Value>  /*dim*/ ) const
  {
    constexpr auto dim = Dimension<Value>{};
    return evolve(state_it, material, dtdh, dim, right_face_select);
  }

  /// Returns the left right state in the forward direction, where the forward
  /// direction is one of:
  ///
  ///   { right (x-dim), up (y-dim), inward (z-dim) }
  /// 
  /// \param[in]  state_it   The state iterator to reconstruct from.
  /// \param[in]  material   The material for the system.
  /// \param[in]  dtdh       The space time scaling factor.
  /// \tparam     Iterator   The type of the iterator, which must be
  ///                        multi-dimensional 
  /// \tparam     Material   The type of the material.
  /// \tparam     T          The type of the scaling factor.
  /// \tparam     Value      The value which defines the dimension.
  template <typename Iterator, typename Material, typename T, std::size_t Value>
  fluidity_host_device constexpr auto
  input_fwrd_right(Iterator&&       state_it,
                   Material&&       material,
                   T                dtdh    ,
                   Dimension<Value> /*dim*/ ) const
  {
    constexpr auto dim = Dimension<Value>{};
    return evolve(state_it.offset(right_face, dim),
                  material                        ,
                  dtdh                            ,
                  dim                             ,
                  left_face_select                );
  }

  /// Returns the left input state in the backward direction, where the backward
  /// direction is one of:
  ///
  ///   { left (x-dim), down (y-dim), outward (z-dim) }
  /// 
  /// \param[in]  state_it   The state iterator to reconstruct from.
  /// \param[in]  material   The material for the system.
  /// \param[in]  dtdh       The space time scaling factor.
  /// \tparam     Iterator   The type of the iterator, which must be
  ///                        multi-dimensional 
  /// \tparam     Material   The type of the material.
  /// \tparam     T          The type of the scaling factor.
  /// \tparam     Value      The value which defines the dimension.
  template <typename Iterator, typename Material, typename T, std::size_t Value>
  fluidity_host_device constexpr auto
  input_back_left(Iterator&&       state_it,
                  Material&&       material,
                  T                dtdh    ,
                  Dimension<Value> /*dim*/ ) const
  {
    constexpr auto dim = Dimension<Value>{};
    return evolve(state_it.offset(left_face, dim),
                  material                       ,
                  dtdh                           ,
                  dim                            ,
                  right_face_select              ); 
  }

  /// Returns the right state in the backward direction, where the backward
  /// direction is one of:
  ///
  ///   { left (x-dim), down (y-dim), outward (z-dim) }
  /// 
  /// \param[in]  state_it   The state iterator to reconstruct from.
  /// \param[in]  material   The material for the system.
  /// \param[in]  dtdh       The space time scaling factor.
  /// \tparam     Iterator   The type of the iterator, which must be
  ///                        multi-dimensional 
  /// \tparam     Material   The type of the material.
  /// \tparam     T          The type of the scaling factor.
  /// \tparam     Value      The value which defines the dimension.
  template <typename Iterator, typename Material, typename T, std::size_t Value>
  fluidity_host_device constexpr auto
  input_back_right(Iterator&&       state_it,
                   Material&&       material,
                   T                dtdh    ,
                   Dimension<Value> /*dim*/ ) const
  {
    constexpr auto dim = Dimension<Value>{};
    return evolve(state_it, material, dtdh, dim, left_face_select);
  }
};

}} // namespace fluid::recon

#endif // FLUIDITY_RECONSTRUCTION_MUSCL_RECONSTRUCTOR_HPP