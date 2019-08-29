//==--- fluidity/material/material_iterator.hpp ------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  material_iterator.hpp
/// \brief This file defines an iterator class for a material, which combines
///        the iterators for the levelset and state data along with the equation
///        of state for the material.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_MATERIAL_MATERIAL_ITERATOR_HPP
#define FLUIDITY_MATERIAL_MATERIAL_ITERATOR_HPP

#include <fluidity/utility/portability.hpp>

namespace fluid    {
namespace material {

/// This is a wrapper class which can be used to pass the properties of a
/// material to kernels so that data for the materials can be accessed easily in
/// the kernels. 
//
/// The MaterialIterator class combines iterators for state and levelset data
/// for a material along with its equation of state. This allows the underlying
/// data storage for the iterator data to be stored in the most efficient
/// format, but to have class-like syntax when using the material iterator.
/// \tparam EquationOfState   The equation of state for the material.
/// \tparam LevelsetIterator  The type of the iterator over the levelset data.
/// \tparam StateIterator     The type of the iterator over the state data.
template <
  typename EquationOfState ,
  typename LevelsetIterator,
  typename StateIterator
>
class MaterialIterator {
 public:
  /// The type of the equation of state for the material.
  using eos_t         = std::decay_t<EquationOfState>;
  /// The type of the levelset iterator being wrapped.
  using levelset_it_t = std::decay_t<LevelsetIterator>;
  /// The type of the state iterator being wrapped.
  using state_it_t    = std::decay_t<StateIterator>;
  /// The type of the state data which is iterated over.
  using state_t       = typename state_it_t::value_t;
  /// Defines the type of the execution policy for the iterator.
  using exec_t        = typename state_it_t::exec_t;

  /// Defines the number of dimensions for the iterators.
  static constexpr auto dimensions = state_it_t::dimensions;

 private:
  /// Defines the type of the material iterator.
  using self_t = MaterialIterator;

  eos_t         _eos;       //!< The equation of state. 
  levelset_it_t _ls_it;     //!< An iterator over the material levelset data.
  state_it_t    _state_it;  //!< An iterator over the material state data.

  /// Checks that the levelset and state iterators have the same number of
  /// dimensions.
  fluidity_host_device constexpr void assert_dimensionality_match() const {
    static_assert(
      levelset_it_t::dimensions == state_it_t::dimensions,
      "Levelset and state iterators must have the same dimensionality"
    );
  }

 public:
  /// Constructor to create the material iterator, for a material with an
  /// equation of state defined by \p eos. The iterator iterates over the
  /// \p ls_it levelset iterator and the \p state_it state data.
  /// \param[in] eos        The equation of state for the material.
  /// \param[in] ls_it      An iterator over the material levelset data.
  /// \param[in] state_it   An iterator over the material state data.
  /// \tparam    EqOfState  The type of the equation of state.
  /// \tparam    LsIterator The type of the levelset iterator.
  /// \tparam    StIterator The type of the state iterator.
  template <typename EqOfState, typename LsIterator, typename StIterator>
  MaterialIterator(EqOfState&& eos, LsIterator&& ls_it, StIterator&& state_it) : 
  _eos(std::forward<EqOfState>(eos))          ,
  _ls_it(std::forward<LsIterator>(ls_it))     ,
  _state_it(std::forward<StIterator>(state_it)) {
    assert_dimensionality_match();
  }

  //==--- [Aaccess] --------------------------------------------------------==//

  /// Returns a reference to the equation of state for the material iterator.
  fluidity_host_device auto& eos() {
    return _eos;
  }

  /// Returns a const reference to the equation of state for the material
  /// iterator.
  fluidity_host_device auto& eos() const {
    return _eos;
  }

  /// Returns an iterator to the levelset data.
  fluidity_host_device auto& levelset_iterator() const {
    return _ls_it;
  }

  /// Returns an iterator to the state data.
  fluidity_host_device auto& state_iterator() const {
    return _state_it;
  }

  /// Returns an iterator to the levelset data.
  fluidity_host_device auto& levelset_iterator() {
    return _ls_it;
  }

  /// Returns an iterator to the state data.
  fluidity_host_device auto& state_iterator() {
    return _state_it;
  }

  /// Returns a reference to the levelset data currently pointed to by the
  /// levelset iterator for the material iterator.
  fluidity_host_device auto& levelset() {
    return *_ls_it;
  }

  /// Returns the value of the levelset data currently pointed to by the
  /// levelset iterator for the material iterator.
  fluidity_host_device auto levelset() const {
    return *_ls_it;
  }

  /// Returns a reference to the state data currently pointed to by the
  /// state iterator for the material iterator.
  fluidity_host_device auto& state() {
    return *_state_it;
  }

  /// Returns a copy of the state data currently pointed to by the state
  /// iterator for the material iterator.
  fluidity_host_device auto state() const {
    return *_state_it;
  }

  //==--- [Size] -----------------------------------------------------------==//
  
  /// Returns the size of the iterator -- the total number of elements in the
  /// domain to iterate over.
  constexpr auto size() const {
    return _state_it.size();
  }

  /// Returns the size of the iterable data for the \p dim dimension.
  /// \param[in] dim The dimension to get the size of.
  /// \param[in] Dim The type of the dimension specifier.
  template <typename Dim>
  constexpr auto size(Dim&& dim) const {
    return _state_it.size(dim);
  }

  //==--- [Shifting/offsetting] --------------------------------------------==//

  /// Offsets the iterators in the \p dim dimension by the \p amount, returning
  /// a new material iterator.
  /// \param[in] amount The amount to offset the iterators by.
  /// \param[in] dim    The dimension to offset the iterators in.
  /// \tparam    Dim    The type of the dimension specifier.
  template <typename Dim>
  fluidity_host_device auto offset(int amount, Dim&& dim) const -> self_t {
    return self_t{
      _eos                         , 
      _ls_it.offset(amount, dim)   ,
      _state_it.offset(amount, dim)
    };
  }

  /// Shifts the iterators in the \p dim dimension by the \p amount.
  /// \param[in] amount The amount to shift the iterators by.
  /// \param[in] dim    The dimension to shift the iterators in.
  /// \tparam    Dim    The type of the dimension specifier.
  template <typename Dim>
  fluidity_host_device auto shift(int amount, Dim&& dim) -> void {
    _ls_it.shift(amount, dim);
    _state_it.shift(amount, dim);
  }
};

//===--- [Creation] --------------------------------------------------------==//

/// Utility function to make a material iterator. This is the preferred way for
/// creating a material iterator since it infers the types of the equation of
/// state, levelset iterator, and state iterator.
/// \param[in] eos              The equation of state for the material.
/// \param[in] ls_it            An iterator over the material levelset data.
/// \param[in] state_it         An iterator over the material state data.
/// \tparam    EquationOfState  The type of the equation of state.
/// \tparam    LevelsetIterator The type of the levelset iterator.
/// \tparam    StateIterator    The type of the state iterator.
template <
  typename EquationOfState,
  typename LevelsetIterator,
  typename StateIterator
>
fluidity_host_device auto make_material_iterator(
  EquationOfState&&  eos     ,
  LevelsetIterator&& ls_it   ,
  StateIterator&&    state_it
) {
  using material_iter_t = MaterialIterator<
    std::decay_t<EquationOfState> ,
    std::decay_t<LevelsetIterator>,
    std::decay_t<StateIterator>
  >;
  return material_iter_t(
    std::forward<EquationOfState>(eos)   ,
    std::forward<LevelsetIterator>(ls_it),
    std::forward<StateIterator>(state_it)
  );
}

}} // namespace fluid::material


#endif // FLUIDITY_MATERIAL_MATERIAL_ITERATOR_HPP
