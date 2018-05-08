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

#include "riemann_input.hpp"
#include <fluidity/dimension/thread_index.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid {
namespace recon {

/// The MusclHancock struct defines a functor which uses the MUSCL-Hancock
/// scheme to reconstruct a given state vector. The implementation overloads
/// the function call operator to invoke the reconstructor.
/// 
/// The class is templated on the slope limiter so that different versions of
/// the MUSCL reconstructor can be composed.
/// 
/// \tparam SlopeLimter The type of the sloper limiter to use.
template <typename T, typename SlopeLimiter>
struct MHReconstructor {
  /// Defines the type of the slope limiter used by the reconstructor.
  using limiter_t = SlopeLimiter;
  /// Defines the data type used by the slope limiter, and hence the
  /// reconstructor.
  using value_t   = std::decay_t<T>;

  /// Sets the number of elements which are required in the backward and forward
  /// directions during the reconstruction process.
  static constexpr size_t width = 2;

 private:
  /// Defines the value of the reconstruction application to a forward face.
  static constexpr int  forward_face  = 1;
  /// Defiens the value of the reconstruction application to a backward face.
  static constexpr int  backward_face = -1;
  /// Defines a vector with all its values being a quarter.
  static constexpr auto quarter       = value_t{0.25};
  /// Defines a vector with all its values being a half.
  static constexpr auto half          = value_t{0.5};
  /// Defines a vector with all its values being one.
  static constexpr auto one           = value_t{1.0};

 public:
  /// The ReconImpl struct applies reconstruction of a state for a specific face
  /// a single dimension. The face to apply the reconstruction to us defined by
  /// the compile time value Face, where positive indicates faces in the
  /// forward direction, and negative in the backward direction.
  /// \tparam Face  The face to apply reconstruction to.
  template <int Face>
  struct ReconImpl {
    /// Reconstructs the state data in a single direction, returning new data. 
    /// \param[in]  state      The state to reconstruct around.
    /// \param[in]  mat        The material describing the system.
    /// \param[in]  dtdh       The scaling factor: $\frac{dt}{dx}$.
    /// \param[in]  dim        The dimension to update over.
    /// \tparam     Iterator   The type of the state iterator.
    /// \tparam     Material   The type of the material.
    /// \tparam     Value      The value which defines the dimension.
    template <typename Iterator, typename Material, std::size_t Value>
    fluidity_host_device constexpr decltype(auto) 
    operator()(Iterator&&       state  ,
               Material&&       mat    ,
               value_t          dtdh   ,
               Dimension<Value> /*dim*/) const
    {
      using state_t = std::decay_t<decltype(*state)>;
      static_assert(Face == backward_face || Face == forward_face,
                    "Invalid face for reconstruction");

      constexpr auto dim = Dimension<Value>{};
      const auto eita    = half * limiter_t()(state, dim);

      const auto factor = (half * dtdh)
                        * (state_t(*state - eita).flux(mat, dim)
                        -  state_t(*state + eita).flux(mat, dim));

      return Face == forward_face ? state_t(*state + eita) - factor
                                  : state_t(*state - eita) - factor;
    }
  };

  /// Defines the type of the forward reconstructor.
  using fwrd_recon_t = ReconImpl<forward_face>;
  /// Defines the type of the backward reconstructor.
  using back_recon_t = ReconImpl<backward_face>;

  /// Intiializes the constants used by the reconstructor.
  /// \param[in] omega   The scaling factor for the state differences, i.e
  ///                       $ (1 + \omega)\delta_{i - .5}) $ and
  ///                       $ (1 - \omega)\delta_{i + .5}  $
  ///                    and must be in the range [-1, 1].
  fluidity_host_device constexpr MHReconstructor(value_t omega = 0)
  :   _alpha{half * (one + omega)}, _beta{half * (one - omega)}, _omega{omega}
  {
    assert(_omega >= value_t{-1.0} && omega <= value_t{1.0} &&
           "Omega must be in the range [-1.0, 1.0]");
  }

  /// Reconstructs the state data, returning new data. This reconstructs between
  /// the cell defined by \p state and the one in the forward direction, and
  /// returns the two state types which are the reconstructed states in the
  /// forward direction.
  /// 
  /// No direction option is provided because this can be done from the calling
  /// code by first offsetting the state iterator which is the input to this
  /// function, i.e by using ``state = state(-1, dim)`` before the
  /// reconstruction call.
  /// 
  /// \param[in]  state      The state to reconstruct around.
  /// \param[in]  mat        The material describing the system.
  /// \param[in]  dtdh       The scaling factor: $\frac{dt}{dx}$.
  /// \param[in]  dim        The dimension to update over.
  /// \tparam     Iterator   The type of the iterator, which must be a block
  ///                        iterator over a State.
  /// \tparam     Material   The type of the material.
  /// \tparam     Value      The value which defines the dimension.
  template <typename Iterator, typename Material, std::size_t Value>
  fluidity_host_device constexpr decltype(auto) 
  operator()(Iterator&&       state   ,
             Material&&       material,
             value_t          dtdh    ,
             Dimension<Value> /*dim*/ ) const
  {
    using state_t = std::decay_t<decltype(*state)>;

    // Define the dimension to ensure constexpr functionality:
    constexpr auto dim = Dimension<Value>{};
    return make_riemann_input<state_t>(
      fwrd_recon_t{}(state, material, dtdh, dim),
      back_recon_t{}(state.offset(1, dim), material, dtdh, dim)
    );
  }

 private:
  value_t _alpha  = 0;  //!< Backward state scale     : $(1 + \omega)$.
  value_t _beta   = 0;  //!< Forward state scale      : $(1 - \omega)$.
  value_t _omega  = 0;  //!< General scaling factor   : $\omega$.
};

}} // namespace fluid::recon

#endif // FLUIDITY_RECONSTRUCTION_MUSCL_RECONSTRUCTOR_HPP