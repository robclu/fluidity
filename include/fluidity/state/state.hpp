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
#include <fluidity/container/number.hpp>
#include <string>
#include <vector>

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

  /// Returns the format of the state.
  static constexpr FormType    format                = Form;
  /// Returns the number of additional components for the state.
  static constexpr std::size_t additional_components = Components;
  /// Returns the number of dimensions in the state.
  static constexpr std::size_t dimensions            = Dimensions;
  /// Returns the storage layout of the state.
  static constexpr auto        storage_layout        = Format;
  /// Returns the number of elements in the state.
  static constexpr auto        elements              = 2 
                                                     + dimensions
                                                     + additional_components;

  /// The index struct returns the values where data is stored in the state.
  struct index {
    /// Defines that the density is always stored as the first state element.
    static constexpr int density  = 0;
    /// Defines that the presure is the second element if the state is prim.
    static constexpr int pressure = format == FormType::primitive ? 1 : -1;
    /// Defines that the energy is the second element if the stat is cons.
    static constexpr int energy   = format == FormType::conservative ? 1 : -1;
    /// Defines the offset to the first velocity element.
    static constexpr int v_offset = 2;
    /// Defines the offset to the first additional element.
    static constexpr int a_offset = v_offset + dimensions;

    /// Creates an array of names where the index in the array of the name
    /// defines the position of the named element in the state, i.e
    /// 
    /// ~~~
    ///   [ "density", "pressure", "v_x", "v_y", "add_0" ]
    /// ~~~
    /// 
    /// describes the layout of the state data elements.
    static constexpr auto element_names()
    {
      // ASCII offst to char code x:
      constexpr int ascii_offset = 120;

      std::vector<std::string> names = { "density" };
      names.emplace_back(format == FormType::primitive ? "pressure" : "energy");

      unrolled_for<dimensions>([&names] (auto i)
      {
        std::string s = std::string("v_") + char(ascii_offset + i);
        names.push_back(s);
      });
      unrolled_for<additional_components>([&names] (auto i)
      {
        std::string s = "add_" + std::to_string(i);
        names.emplace_back(s);
      });
      return names;
    }

    /// Defines the type of map used to store offsets and their named
    /// equivalents.
    static constexpr int from_name(const char* name)
    {
      auto names = element_names();
      for (std::size_t i = 0; i < names.size(); ++i)
      {
        if (name ==  names[i])
          return i;
      }
      return -1;
    }

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
    static constexpr int velocity(Dimension<Value> /*dim*/)
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
    template <int N>
    static constexpr int additional(Number<N> /*dim*/)
    {
      return Number<N>::value + a_offset;
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
  /// \tparam     V     The value of the dimension.
  template <std::size_t V>
  fluidity_host_device constexpr auto velocity(Dimension<V> /*dim*/) const
  {
    return detail::velocity(*this, Dimension<V>{});
  }

  /// Sets the velocity of the state for a given dimension \p dim.
  /// \param[in] value The value to set the velocity to.
  /// \param[in] dim   The dimension to set the velocity for.
  /// \tparam    V     The value whihc defines the dimension.
  template <std::size_t V>
  fluidity_host_device constexpr void
  set_velocity(value_t value, Dimension<V> /*dim*/)
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
  /// \param[in] n The nth additional component to return.
  fluidity_host_device constexpr auto additional(std::size_t n) const
  {
    return this->operator[](index::additional(n));
  }

  /// Returns the \p nth additional component of the state, if it exists. This
  /// version is more safe and performant as it allows compile time checking of
  /// the validity of the requested component (i.e that it exists). The
  /// offsetting of the index is also computed at compile time.
  /// 
  /// Example usage is:
  /// 
  /// ~~~cpp
  /// unrolled_for<state_t::additional_components>([&] (auto i)
  /// {
  ///   auto component = state.additional(Number<i>{});
  ///   do_something(compoenent);
  /// }
  /// ~~~
  /// 
  /// \param[in] i      The Number type which wraps the index of the additional
  ///            type to get.
  /// \tparam    Index  The compile time value of the index to get.
  template <int Index>
  fluidity_host_device constexpr auto additional(Number<Index> i) const
  {
    static_assert(Index < additional_components,
                  "Out of range additional component access!");
    return this->operator[](index::additional(Number<Index>{}));
  }

  /// Sets the value of the \p index additional component of the state to
  /// \p value.
  /// \param[in] index The index of the element to set.
  /// \param[in] value The value to set the component to.
  fluidity_host_device constexpr void
  set_additional(value_t value, std::size_t index)
  {
    this->operator[](index::additional(index)) = value;
  }

  /// Sets the value of the \p index additional component of the state to
  /// \p value. THis s the more performant version as it allows compile time
  /// assertation for the validity of the requested index.
  /// \param[in] index The index of the element to set.
  /// \param[in] value The value to set the component to.
  /// \tparam    Index The value of the index of the compoenent.
  template <int Index>
  fluidity_host_device constexpr void
  set_additional(value_t value, Number<Index> index)
  {
    static_assert(Index < additional_components,
                  "Out of range additional component access!");
    this->operator[](index::additional(Number<Index>{})) = value;
  }

  /// Returns a conservative form of the state, regardless of whether the
  /// state is primitive or conservative.
  /// \param[in] material The material to use in the case where conversion is
  ///            required.
  /// \tparam    Material The type of the material.
  template <typename Material>
  fluidity_host_device constexpr auto conservative(Material&& material) const
  {
    return detail::conservative(*this, std::forward<Material>(material));
  }

  /// Returns a primitive form of the state, regardless of whether the
  /// state is primitive or conservative.
  /// \param[in] material The material to use in the case where conversion is
  ///            required.
  /// \tparam    Material The type of the material.
  template <typename Material>
  fluidity_host_device constexpr auto primitive(Material&& material) const
  {
    return detail::primitive(*this, std::forward<Material>(material));
  }


  /// Returns the flux from the state vector, in terms of a specific spacial
  /// dimension \p dim for a given material \p material.
  /// \param[in] material The material which describes the system.
  /// \param[in] dim      The dimension to compute the fluxes in terms of.
  /// \tparam    Material The type of the material.
  /// \tparam    Value    The value which defines the dimension.
  template <typename Material, std::size_t Value>
  fluidity_host_device constexpr auto
  flux(Material&& material, Dimension<Value> /*dim*/) const
  {
    constexpr auto dim = Dimension<Value>{};
    return detail::flux(*this, std::forward<Material>(material), dim);
  }

  /// Returns the sum of sqaured velocities for the state.
  fluidity_host_device constexpr value_t v_squared_sum() const
  {
    return detail::v_squared_sum(*this);
  }

  /// Returns the number of elements (total components) of the state.
  fluidity_host_device constexpr auto size() const
  {
    return index::a_offset + additional_components;
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