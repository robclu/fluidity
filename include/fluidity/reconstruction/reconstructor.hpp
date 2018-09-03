//==--- fluidity/reconstruction/reconstructor.hpp- -----------*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  reconstructor.hpp
/// \brief This file defines the interface to which all reconstructors must
///        conform.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_RECONSTRUCTION_RECONSTRUCTOR_HPP
#define FLUIDITY_RECONSTRUCTION_RECONSTRUCTOR_HPP

#include "reconstructor_traits.hpp"
#include <fluidity/state/state_traits.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid {
namespace recon {

/// The Reconstructor class defines the interface to which all reconstructors
/// must conform. The implementation is provided by the template type.
/// \tparam ReconImpl The type of the reconstruction implementation.
template <typename ReconImpl>
class Reconstructor
{
  /// Defines the type of the reconstructor implementation.
  using impl_t   = ReconImpl;
  /// Defines the type of the traits for the limiter.
  using traits_t = ReconstructorTraits<impl_t>;

  /// Returns a pointer to the implementation.
  fluidity_host_device impl_t* impl()
  {
    return static_cast<impl_t*>(this);
  }

  /// Returns a const pointer to the implementation.
  fluidity_host_device const impl_t* impl() const
  {
    return static_cast<const impl_t*>(this);
  }

 public:
  /// Defines the type of the limiter used for the reconstruction.
  using limiter_t = typename traits_t::limiter_t;
  /// Defines the number of elements required for reconstruction.
  static constexpr auto width = traits_t::width;

  /// Returns the left input state in the forward direction, where the forward
  /// direction is one of:
  ///
  ///   { right (x-dim), up (y-dim), inward (z-dim) }
  /// 
  /// \param[in]  state_it   The state iterator to reconstruct from.
  /// \param[in]  mat        The material describing the fluid/solid.
  /// \param[in]  dtdh       The scaling factor: $\frac{dt}{dh}$.
  /// \param[in]  dim        The dimension to update over.
  /// \tparam     Iterator   A multi-dimensional iterator over the states. 
  /// \tparam     Material   The type of the material.
  /// \tparam     T          The type of the scaling factor.
  /// \tparam     Value      The value which defines the dimension.
  template <typename Iterator, typename Material, typename T, typename Dim>
  fluidity_host_device constexpr auto
  forward_left(Iterator&& state_it,
               Material&& material,
               T          dtdh    ,
               Dim        dim ) const
  {
    assert_valid_iterator<Iterator>();
    return impl()->input_fwrd_left(std::forward<Iterator>(state_it),
                                   std::forward<Material>(material),
                                   dtdh, dim                       );
  }

  /// Returns the left right state in the forward direction, where the forward
  /// direction is one of:
  ///
  ///   { right (x-dim), up (y-dim), inward (z-dim) }
  /// 
  /// \param[in]  state_it   The state iterator to reconstruct from.
  /// \param[in]  mat        The material describing the fluid/solid.
  /// \param[in]  dtdh       The scaling factor: $\frac{dt}{dh}$.
  /// \param[in]  dim        The dimension to update over.
  /// \tparam     Iterator   The type of the iterator, which must be
  ///                        multi-dimensional 
  /// \tparam     Material   The type of the material.
  /// \tparam     T          The type of the scaling factor.
  /// \tparam     Value      The value which defines the dimension.
  template <typename Iterator, typename Material, typename T, typename Dim>
  fluidity_host_device constexpr auto
  forward_right(Iterator&& state_it,
                Material&& material,
                T          dtdh    ,
                Dim        dim     ) const
  {
    assert_valid_iterator<Iterator>();
    return impl()->input_fwrd_right(std::forward<Iterator>(state_it),
                                    std::forward<Material>(material),
                                    dtdh, dim                       );
  }

  /// Returns the left input state in the backward direction, where the backward
  /// direction is one of:
  ///
  ///   { left (x-dim), down (y-dim), outward (z-dim) }
  /// 
  /// \param[in]  state_it   The state iterator to reconstruct from.
  /// \param[in]  mat        The material describing the fluid/solid.
  /// \param[in]  dtdh       The scaling factor: $\frac{dt}{dh}$.
  /// \param[in]  dim        The dimension to update over.
  /// \tparam     Iterator   The type of the iterator, which must be
  ///                        multi-dimensional 
  /// \tparam     Material   The type of the material.
  /// \tparam     T          The type of the scaling factor.
  /// \tparam     Value      The value which defines the dimension.
  template <typename Iterator, typename Material, typename T, typename Dim>
  fluidity_host_device constexpr auto
  backward_left(Iterator&& state_it,
                Material&& material,
                T          dtdh    ,
                Dim        dim     ) const
  {
    assert_valid_iterator<Iterator>();
    return impl()->input_back_left(std::forward<Iterator>(state_it),
                                   std::forward<Material>(material),
                                   dtdh, dim                       );
  }

  /// Returns the right state in the backward direction, where the backward
  /// direction is one of:
  ///
  ///   { left (x-dim), down (y-dim), outward (z-dim) }
  /// 
  /// \param[in]  state_it   The state iterator to reconstruct from.
  /// \param[in]  mat        The material describing the fluid/solid.
  /// \param[in]  dtdh       The scaling factor: $\frac{dt}{dh}$.
  /// \param[in]  dim        The dimension to update over.
  /// \tparam     Iterator   The type of the iterator, which must be
  ///                        multi-dimensional 
  /// \tparam     Material   The type of the material.
  /// \tparam     T          The type of the scaling factor.
  /// \tparam     Value      The value which defines the dimension.
  template <typename Iterator, typename Material, typename T, typename Dim>
  fluidity_host_device constexpr auto
  backward_right(Iterator&& state_it,
                 Material&& material,
                 T          dtdh    ,
                 Dim        dim     ) const
  {
    assert_valid_iterator<Iterator>();
    return impl()->input_back_right(std::forward<Iterator>(state_it),
                                    std::forward<Material>(material),
                                    dtdh, dim                       );
  }

 private:
  /// Checks that the iterator iterates over constervative state data.
  /// \tparam Iterator The type of the iterator to check the validity of.
  template <typename Iterator>
  fluidity_host_device static constexpr void assert_valid_iterator()
  {
/*
    using iter_t  = std::decay_t<Iterator>;
    using value_t = typename iter_t::value_t;
    static_assert(state::traits::is_conservative_v<value_t>,
      "Can only reconstruct using an iterator which iterates over states that"
      "are in conservative form. Please change iterator to iterate over"
      "conservative state data.");

    static_assert(iter_t::is_multi_dimensional,
      "Can only reconstruct using an iterator which is multi dimensional."
      "Convert current iterator to a multi dimensional version.");
*/
  }
};

}} // namespace fluid::recon

#endif // FLUIDITY_RECONSTRUCTION_RECONSTRUCTOR_HPP