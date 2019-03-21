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
template < typename      T
         , FormType      Form         
         , std::size_t   Dimensions
         , std::size_t   Components
         , StorageFormat Format>
class State : public traits::storage_t<T, Dimensions, Components, Format> {
 private:
  /// Defines the type of a dispatch tag for the state.
  using dispatch_tag_t = traits::detail::StateDispatchTag<Form>;
 public:
  /// Defines an alias for the type of storage;
  using storage_t = traits::storage_t<T, Dimensions, Components, Format>;
  /// Defines the type of the data elements in the state.
  using value_t   = std::decay_t<T>;
  /// Defines the type of this state.
  using self_t    = State;

  /// Defines a valid enable type if the template type is not the value_t type.
  /// \tparam TT The template paramter to check the type of.
  template <typename TT>
  using non_value_enable_t = std::enable_if_t<
    !std::is_convertible<std::decay_t<TT>, value_t>::value, int>;
  /// Defines a valid type if the template type is convertible to the valuet_t
  /// type.
  /// \tparam TT The template paramter to check the type of.
  template <typename TT>
  using value_enable_t = std::enable_if_t<
    std::is_convertible<std::decay_t<TT>, value_t>::value, int>;

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

 public:
  /// Use the storage class constructors.
  using storage_t::storage_t;

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
    static auto element_names()
    {
      // ASCII offst to char code x:
      constexpr int ascii_offset = 120;

      std::vector<std::string> names = { "rho" };
      names.emplace_back(format == FormType::primitive ? "p" : "E");

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
    /// \param[in] dim    The dimensions to get the velocity index for.
    /// \tparam    Value  The value of the compile time dimension type. 
    template <typename Dim>
    static constexpr int velocity(Dim dim)
    {
      return dim + v_offset;
    }

    /// Returns the offset to the Nth additional element of the state.
    /// \param[in] n  The index of the element to get.
    /// \tparam    N  The value of the index of the element to get.
    template <typename Index>
    static constexpr int additional(Index index)
    {
      return index + a_offset;
    }
  };

  /// Constructor to create the state from a container. This does not check that
  /// the container is the same size as the state.
  /// 
  /// TODO: Add debug mode size checking ...
  /// 
  /// \param[in] container The container with the data to set the elements to.
  /// \tparam    Container The type of the container.
  template <typename Container, non_value_enable_t<Container> = 0>
  fluidity_host_device State(Container&& container)
  {
    unrolled_for_bounded<elements>([&] (auto i)
    {
      this->operator[](i) = container[i];
    });
  }

  /// Constructor to create a state with a default value for each element.
  /// \param[in] value The value to initialize each element to.
  template <typename Value, value_enable_t<Value> = 0>
  fluidity_host_device State(Value&& value)
  {
    unrolled_for_bounded<elements>([&] (auto i)
    {
      this->operator[](i) = value;
    });
  }

  /// Constructor to fill the state with a list of data.
  /// \param[in] values The values of the data for the state.
  /// \tparam    Values The types of the values.
  template <typename... Values>
  fluidity_host_device State(Values&&... values)
  : storage_t(std::forward<Values>(values)...) {}

  //==--- Component setting ------------------------------------------------==//

  /// Sets the data for the density component for the state.
  /// \param[in] rho The density component to use to set the density.
  void set_component(components::density_t rho)
  {
    set_density(rho.value);
  }

  /// Sets the data for the pressure component for the state.
  /// \param[in] pressure The pressure component to use to set the pressure.
  void set_component(components::pressure_t p)
  {
    if (format == FormType::primitive)
      set_pressure(p.value);
  }

  /// Sets the data for the x-velocity component for the state.
  /// \param[in] vx The velocity component to use to set the x velocity.
  void set_component(components::v_x_t vx)
  {
    if (dimensions >= 1)
      set_velocity(vx.value, std::size_t{0});
  }

  /// Sets the data for the y-velocity component for the state.
  /// \param[in] vy The velocity component to use to set the y velocity.
  void set_component(components::v_y_t vy)
  {
    if (dimensions >= 2)
      set_velocity(vy.value, std::size_t{1});
  }

  /// Sets the data for the z-velocity component for the state.
  /// \param[in] vz The velocity component to use to set the z velocity.
  void set_component(components::v_z_t vz)
  {
    if (dimensions >= 3)
      set_velocity(vz.value, std::size_t{2});
  }

  //==--- Data access ------------------------------------------------------==//

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
  /// \param[in]  mat  The material for the system the state represents.
  /// \tparam     Mat  The type of the material.
  template <typename Mat>
  fluidity_host_device constexpr auto pressure(Mat&& mat) const
  {
    return detail::pressure(*this, std::forward<Mat>(mat));
  }

  /// Sets the pressure of the state, if the state is primitive. This overload
  /// is only enabled for primitive form states.
  /// other conservative components.
  /// \param[in] value The value to set the pressure for the state to.
  fluidity_host_device constexpr void set_pressure(value_t value)
  {
    set_pressure_impl(value, traits::state_dispatch_tag<self_t>);
  }

  /// Returns the internal energy of the state.
  /// \param[in]  mat   The material for the system the state represents.
  /// \tparam     Mat   The type of the material.
  template <typename Mat>
  fluidity_host_device constexpr auto energy(Mat&& mat) const
  {
    return detail::energy(*this, std::forward<Mat>(mat));
  }

  /// Sets the energy of the state. This overload is enabled for conservative
  /// form states.
  /// \param[in] value The value to set the energy to.
  fluidity_host_device constexpr void set_energy(value_t value) 
  {
    set_energy_impl(value, traits::state_dispatch_tag<self_t>);
  }

  /// Returns the velocity in a given dimension (direction). Asking for the
  /// velocity for a dimension which is not valid for the state (i.e the y
  /// velocity for a 1D state, is undefined behaviour).
  /// \param[in]  dim   The dimension to get the velocity for.
  /// \tparam     Dim   The type of the dimension.
  template <typename Dim>
  fluidity_host_device constexpr auto velocity(Dim dim) const
  {
    return detail::velocity(*this, dim);
  }

  /// Sets the velocity of the state for a given dimension \p dim, when the
  /// state has a primitive form.
  /// \param[in] value The value to set the velocity to.
  /// \param[in] dim   The dimension to set the velocity for.
  /// \tparam    Dim   The type of the dimension.
  template <typename Dim>
  fluidity_host_device constexpr void set_velocity(value_t value, Dim dim)
  {
    set_velocity_impl(value, dim, traits::state_dispatch_tag<self_t>);
  }

  /// Returns the \p ith additional component of the state, if it exists.
  /// \param[in] i     The ith additional component to return. If \p i is not a
  /// valid index for an additional element then the behaviour is undefined.
  /// \tparam    Index The type of the index for the additional element.
  template <typename Index>
  fluidity_host_device constexpr auto additional(Index i) const
  {
    return this->operator[](index::additional(i));
  }

  /// Sets the value of the \p i additional component of the state to \p value.
  /// If \p i is not a valid index for an additional element, the behaviour is
  /// undefined.
  /// \param[in] i     The index of the element to set.
  /// \param[in] value The value to set the component to.
  /// \tparam    Index The type of the index.
  template <typename Index>
  fluidity_host_device constexpr void set_additional(value_t value, Index i)
  {
    this->operator[](index::additional(i)) = value;
  }

  /// Returns a conservative form of the state, regardless of whether the
  /// state is primitive or conservative.
  /// \param[in] mat The material to use when conversion is required.
  /// \tparam    Mat The type of the material.
  template <typename Mat>
  fluidity_host_device constexpr auto conservative(Mat&& mat) const
  {
    return detail::conservative(*this, std::forward<Mat>(mat));
  }

  /// Returns a primitive form of the state, regardless of whether the
  /// state is primitive or conservative.
  /// \param[in] mat The material to use when conversion is required.
  /// \tparam    Mat The type of the material.
  template <typename Mat>
  fluidity_host_device constexpr auto primitive(Mat&& mat) const
  {
    return detail::primitive(*this, std::forward<Mat>(mat));
  }

  /// Returns the flux for the state vector, in terms of a specific spacial
  /// dimension \p dim for a given material \p mat.
  /// \param[in] mat  The material which describes the system.
  /// \param[in] dim  The dimension to compute the fluxes in terms of.
  /// \tparam    Mat  The type of the material.
  /// \tparam    Dim  The value which defines the dimension.
  template <typename Mat, typename Dim>
  fluidity_host_device constexpr auto flux(Mat&& mat, Dim dim) const
  {
    return detail::flux(*this, std::forward<Mat>(mat), dim);
  }

  /// Returns the maximum wavespeed, $S_{max}$ of the state, which is defined
  /// as:
  /// \begin{equation}
  ///   S_{max} = \max_i \{ |u| + a }
  /// \end{equation}
  /// where $u_i$ is velocity for the state in dimension $i$, and $a$ is the
  /// speed of sound in the \p mat material.
  /// \param[in] mat The material for the system.
  /// \tparam    Mat The type of the material.
  template <typename Mat>
  fluidity_host_device constexpr value_t max_wavespeed(Mat&& mat)
  {
    return detail::max_wavespeed(*this, std::forward<Mat>(mat));
  };

  /// Returns the sum of the sqaure of each of the velocity components, $u_i$,
  /// for the state:
  /// \begin{equation}
  ///   \sum_{i}^{N} u_i^2
  /// \end{equation}
  /// where $N$ is the number of spacial dimension for the state.
  fluidity_host_device constexpr value_t v_squared_sum() const
  {
    return detail::v_squared_sum(*this);
  }

  /// Returns the number of elements (total components) of the state.
  fluidity_host_device constexpr auto size() const
  {
    return index::a_offset + additional_components;
  }

 private:
  /// Defines the type of a primitive dispatch tag.
  using prim_tag_t = typename traits::primitive_tag_t;
  /// Defines the type of a conservative dispatch tag.
  using cons_tag_t = typename traits::conservative_tag_t;

  /// Sets the pressure of the state. This overload is only enabled for
  /// primitive form states.
  /// \param[in] v The value to set the pressure for the state to.
  fluidity_host_device constexpr void set_pressure_impl(value_t v, prim_tag_t)
  {
    this->operator[](index::pressure) = v;
  }

  /// Sets the pressure of the state. This overload causes a compile time error
  /// since conservative states do not have a pressure component.
  fluidity_host_device constexpr void set_pressure_impl(value_t, cons_tag_t)
  {
    static_assert(traits::is_primitive_v<self_t>,
                  "Cannot set the pressure for a conservative form state!");
  }

  /// Sets the energy of the state. This overload is causes a compile time error
  /// since primitive states do not have an energy component.
  fluidity_host_device constexpr void set_energy_impl(value_t, prim_tag_t) 
  {
    static_assert(traits::is_conservative_v<self_t>,
                  "Cannot set the energy for a primitive form state!");
  }

  /// Sets the energy of the state. This overload is only enabled for
  /// conservative form states.
  /// \param[in] v The value to set the energy to.
  fluidity_host_device constexpr void set_energy_impl(value_t v, cons_tag_t) 
  {
    this->operator[](index::energy) = v;
  }

  /// Sets the velocity of the state for a given dimension \p dim. This overload
  /// is enabled for primitive states.
  /// \param[in] v     The value to set the velocity to.
  /// \param[in] dim   The dimension to set the velocity for.
  /// \tparam    Dim   The type of the dimension.
  template <typename Dim>
  fluidity_host_device
  constexpr void set_velocity_impl(value_t v, Dim dim, prim_tag_t)
  {
    this->operator[](index::velocity(dim)) = v;
  }

  /// Sets the velocity of the state for a given dimension \p dim. This overload
  /// is enabled for conservative form states.
  /// \param[in] v     The value to set the velocity to.
  /// \param[in] dim   The dimension to set the velocity for.
  /// \tparam    Dim   The type of the dimension.
  template <typename Dim>
  fluidity_host_device
  constexpr void set_velocity_impl(value_t v, Dim dim, cons_tag_t)
  {
    v *= this->operator[](index::density);
    this->operator[](index::velocity(dim)) = v;
  }
};

/// Alias for a primitive state.
/// \tparam T          The type of the data for the state.
/// \tparam Dimensions The number of spacial dimensions.
/// \tparam Components The number of additional components.
/// \tparam Format     The storage format for the state.
template <typename      T,
          std::size_t   Dimensions,
          std::size_t   Components = 0,
          StorageFormat Format     = StorageFormat::row_major>
using primitive_t =
  State<T, FormType::primitive, Dimensions, Components, Format>;

/// Alias for a conservative state.
/// \tparam T          The type of the data for the state.
/// \tparam Dimensions The number of spacial dimensions.
/// \tparam Components The number of additional components.
/// \tparam Format     The storage format for the state.
template <typename      T,
          std::size_t   Dimensions,
          std::size_t   Components = 0,
          StorageFormat Format     = StorageFormat::row_major>
using conservative_t =
  State<T, FormType::conservative, Dimensions, Components, Format>;

} // namespace state
} // namespace fluid

#endif // FLUIDITY_STATE_STATE_HPP