//==--- fluidity/reconstruction/reconstructor_traits.hpp- ----*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  reconstructor_traits.hpp
/// \brief This file defines type traits for reconstructors.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_RECONSTRUCTION_RECONSTRUCTOR_TRAITS_HPP
#define FLUIDITY_RECONSTRUCTION_RECONSTRUCTOR_TRAITS_HPP

namespace fluid {
namespace recon {

/// Forward declarations:
template <typename Limiter> struct MHReconstructor;

/// The ReconstructorTraits class defines traits of reconstructors.
/// \tparam Reconstructor The reconstructor to get the traits for.
template <typename Reconstructor>
struct ReconstructorTraits
{
  /// Defines the type of the reconstructor implementation.
  using reconstructor_t = std::decay_t<Reconstructor>;
  /// Defines the type of the limiter used by the reconstructor.
  using limiter_t       = typename reconstructor_t::limiter_t;
  
  /// Defines the width of the limiter (the number of elements to the side of a
  /// state which are required to perform the limiting).
  static constexpr auto width = reconstructor_t::width;
};

//== Forward declarations and specializations ------------------------------==//

/// Forward declaration of the Muscl-Hancock reconstructor.
/// \tparam Limiter The limiter to use in the reconstruction.
template <typename Limiter> struct MHReconstructor;

/// Specialization for reconstruction traits for the Muscl Hancock
/// reconstructor. 
/// \tparam Limiter The limiter to use in the reconstruction.
template <typename Limiter>
struct ReconstructorTraits<MHReconstructor<Limiter>>
{
  /// Defines the type of the reconstructor implementation.
  using reconstructor_t = MHReconstructor<Limiter>;
  /// Defines the type of the limiter used by the reconstructor.
  using limiter_t       = Limiter;
  
  /// Defines the width of the limiter (the number of elements to the side of a
  /// state which are required to perform the limiting).
  static constexpr auto width = limiter_t::width;
};

}} // namespace fluid::recon

#endif // FLUIDITY_RECONSTRUCTION_RECONSTRUCTOR_TRAITS_HPP