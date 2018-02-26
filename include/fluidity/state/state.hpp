//==--- fluidity/state/state.hpp --------------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  state.hpp
/// \brief This file defines a generic state class.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_STATE_STATE_HPP
#define FLUIDITY_STATE_STATE_HPP

#include "state_impl.hpp"

namespace fluid  {
namespace state  {

/// Defines a class to represent a state.
/// \tparam T The type of data used by the state.
template <typename T,
          FormType      Form,         
          std::size_t   Dimensions,
          std::size_t   Components,
          StorageFormat Format    >
class State : public traits::storage_t<T, Dimensions, Components, Format> {
 public:
  /// Defines an alias for the type of storage;
  using storage_t = traits::storage_t<T, Dimensions, Components, Format>;
  /// Defines the type of the data elements in the state.
  using value_t   = std::decay_t<T>;

  /// The index struct returns the values where data is stored in the state.
  struct index {
    /// Defines that the density is always stored as the first state element.
    static constexpr int density  = 0;
    /// Defines that the presure is the second element if the state is prim.
    static constexpr int pressure = Form == FormType::primitive ? 1 : -1;
    /// Defines that the energy is the second element if the stat is cons.
    static constexpr int energy   = Form == FormType::conservative ? 1 : -1;
    /// Defines the offset to the first velocity element.
    static constexpr int v_offset = 2;
    /// Defines the offset to the first additional element.
    static constexpr int a_offset = v_offset + Components;

    /// Returns the offset to the velocity element to the \p dim direction.
    /// \param[in] dim The dimensions to get the velocity index for.
    static constexpr int velocity(std::size_t dim)
    {
      return dim + v_offset;
    }

    /// Returns the offset to the velocity element to the \p dim direction.
    /// \param[in] dim    The dimensions to get the velocity index for.
    /// \tparam    Value  The value of the compile time dimension type. 
    template <std::size_t Value>
    static constexpr int velocity(Dimension<Value> dim)
    {
      return Dimension<Value>::value + v_offset;
    }

    /// Returns the offset to the \p nth additional element.
    /// \param[in] n The index of the element to get.
    static constexpr int additional(std::size_t n)
    {
      return n + a_offset;
    }

    /// Returns the offset to the Nth additional element of the state.
    /// \param[in] n  The index of the element to get.
    /// \tparam    N  The value of the index of the element to get.
    template <std::size_t N>
    static constexpr int velocity(Index<N> dim)
    {
      return Index<N>::value + a_offset;
    }
  };

  /// Returns the density of the state.
  fluidity_host_device constexpr auto density() const
  { 
    return detail::density(*this);
  }

  /// Sets the density of the state.
  /// \param[in] value The value of the density to set the state to.
  fluidity_host_device constexpr void set_density(value_t value)
  {
    this->operator[](index::density) = value;
  }

  /// Returns the pressure of the state.
  /// \param[in]  material  The material for the system the state represents.
  /// \tparam     Material  The type of the material.
  template <typename Material>
  fluidity_host_device constexpr auto pressure(Material&& material) const
  {
    return detail::pressure(*this, std::forward<Material>(material));
  }

  /// Sets the pressure of the state, if the state is primitive. If this is
  /// called on a conservative form state then a compiler error is generated as
  /// conservative states do not store the pressure, but compute it from the
  /// other conservative components.
  /// \param[in] value The value to set the pressure for the state to.
  fluidity_host_device constexpr void set_pressure(value_t value)
  {
    if constexpr (Form == FormType::primitive)
    {
      this->operator[](index::pressure) = value;
    }
    else
    {
      static_assert(Form == FormType::primitive,
                    "Cannot set the pressure for a conservative state!");
    }
  }

  /// Returns the internal energy of the state.
  /// \param[in]  material   The material for the system the state represents.
  /// \tparam     Material   The type of the material.
  template <typename Material>
  fluidity_host_device constexpr auto energy(Material&& material) const
  {
    return detail::energy(*this, std::forward<Material>(material));
  }

  /// Sets the energy of the state, if the state is conservative. If this is
  /// called on a primitive form state then a compiler error is generated as
  /// primitive states do not store the pressure, but compute it from the
  /// other primitve components.
  /// \param[in] value The value to set the energy to.
  fluidity_host_device constexpr void set_energy(value_t value)
  {
    if constexpr (Form == FormType::conservative)
    {
      this->operator[](index::energy) = value;
    }
    else
    {
      static_assert(Form == FormType::conservative,
                    "Cannot set the energy for a primitive state!");
    }
  }

  /// Returns the velocity in a given dimension (direction). Asking for the
  /// velocity for a dimension which is not valid for the state (i.e the y
  /// velocity for a 1D state, is undefined behaviour). This method is slightly
  /// more performant as all offsets are computed at compile time due to the
  /// value of the dimension being a compile time constant.
  /// 
  /// \todo Add debug check for invalid dim.
  /// 
  /// \param[in]  dim   The dimension to get the velocity for.
  template <std::size_t V>
  fluidity_host_device constexpr auto velocity(Dimension<V> dim) const
  {
    return detail::velocity(*this, Dimension<V>{});
  }

  /// Sets the velocity of the state for a given dimension \p dim.
  /// \param[in] value The value to set the velocity to.
  /// \param[in] dim   The dimension to set the velocity for.
  template <std::size_t V>
  fluidity_host_device constexpr void
  set_velocity(value_t value, Dimension<V> dim)
  {
    // For the conservative form we need to add the density multiplication
    // because {\rho v} is what is actually stored.
    if constexpr (Form == FormType::conservative)
    {
      value *= this->operator[](index::density);
    }
    this->operator[](index::velocity(Dimension<V>{})) = value;
  }

  /// Returns the \p nth additional component of the state, if it exists.
  fluidity_host_device constexpr auto additional(std::size_t n) const
  {
    return this->operator[](index::)
  }
};

/// Alias for a primitive state.
template <typename T,
          std::size_t   Dimensions,
          std::size_t   Components = 0,
          StorageFormat Format     = StorageFormat::row_major>
using primitive_t =
  State<T, FormType::primitive, Dimensions, Components, Format>;

/// Alias for a conservative state.
template <typename T,
          std::size_t   Dimensions,
          std::size_t   Components = 0,
          StorageFormat Format     = StorageFormat::row_major>
using conservative_t =
  State<T, FormType::conservative, Dimensions, Components, Format>;

} // namespace state
} // namespace fluid

#endif // FLUIDITY_STATE_STATE_HPP