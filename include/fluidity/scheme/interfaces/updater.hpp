//==--- fluidity/scheme/interfaces/updater.hpp ------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  updater.hpp
/// \brief This file defines an interface for an updater -- an object which
///        takes input and output data and sets the output data to an updated
///        values computed using the input data.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_INTERFACES_UPDATER_HPP
#define FLUIDITY_SCHEME_INTERFACES_UPDATER_HPP

#include <fluidity/iterator/iterator_traits.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid  {
namespace scheme {

/// The Evolver class provides an interface to which numerical methods must
/// conform. The specific implementation of the Evolver is defined by template
/// type. The __only__ purpose of the evolver interface is to take input data
/// and to compute the updated value of that input data, and then to use it to
/// set output data to that value.
/// \tparam EvolverImpl The implementation of the evolver interface.
template <typename UpdaterImpl>
class Updater {
  /// Defines the type of the updater implementation.
  using impl_t = UpdaterImpl;

  /// Returns a pointer to the implementation.
  fluidity_host_device impl_t* impl() {
    return static_cast<impl_t*>(this);
  }

  /// Returns a const pointer to the implementation.
  fluidity_host_device const impl_t* impl() const {
    return static_cast<const impl_t*>(this);
  }

 public:
  /// Returns the width required by the evolver. This is the number of cells
  /// on a single side which are required.
  constexpr auto width() const {
    return impl()->width();
  }

  /// Interface to update the \p out_it data using the \p in_it data.
  /// \param[in] in_it        The iterable input data to use to evolve.
  /// \param[in] out_it       The iteratable output data to update.
  /// \param[in] dt           The time resolution to use for the update.
  /// \param[in] dh           The spacial resolution to use for the update.
  /// \tparam    InIterator   The type of the input iterator.
  /// \tparam    OutIterator  The type of the output iterator.
  /// \tparam    T            The type of the timestep and resolution.
  template <typename InIterator, typename OutIterator, typename T>
  fluidity_host_device auto
  update(InIterator&& in_it, OutIterator&& out_it, T dt, T dh) const -> void {
    static_assert(
      is_multidim_iter_v<InIterator>, 
      "Input iterator must be a multidimensional iterator!"
    );
    static_assert(
      is_multidim_iter_v<OutIterator>, 
      "Output iterator must be a multidimensional iterator!"
    );

    impl()->update_impl(
      std::forward<InIterator>(in_it)  ,
      std::forward<OutIterator>(out_it),
      dt                               ,
      dh                         
    );
  }

  /// Implemenation of the function to update the \p out_it data using the
  /// \p in_it data and the evaluator, and an additional functor or iterator
  /// over additional data (\p func_or_it).
  ///
  /// \param[in] in_it       The iterable input data to use to evolve.
  /// \param[in] out_it      The iteratable output data to update.
  /// \param[in] dt          The time resolution to use for the update.
  /// \param[in] dh          The spacial resolution to use for the update.
  /// \param[in] func_or_it  A functor which is used in the update.
  /// \param[in] args        Additional arguments for the functor.
  /// \tparam    InIterator  The type of the input iterator.
  /// \tparam    OutIterator The type of the output iterator.
  /// \tparam    T           The type of the timestep and resolution.
  /// \tparam    FuncOrIt    The type of the functor/additional iterator.
  /// \tparam    Args        The types of the additional functor arguments.
  template <
    typename    InIterator ,
    typename    OutIterator,
    typename    T          ,
    typename    FuncOrIt   ,
    typename... Args
  >
  fluidity_host_device auto update(
    InIterator&&  in_it     ,
    OutIterator&& out_it    ,
    T             dt        ,
    T             dh        ,
    FuncOrIt&&    func_or_it,
    Args&&...     args
  ) const -> void {
    static_assert(
      is_multidim_iter_v<InIterator>,
      "Input iterator must be a multidimensional iterator!"
    );
    static_assert(
      is_multidim_iter_v<OutIterator>,
      "Output iterator must be a multidimensional iterator!"
    );

    impl()->update_impl(
      std::forward<InIterator>(in_it)   ,
      std::forward<OutIterator>(out_it) ,
      dt                                ,
      dh                                ,
      std::forward<FuncOrIt>(func_or_it),
      std::forward<Args>(args)...
    );
  }
};

//==--- Traits -------------------------------------------------------------==//

/// Returns true if the type T conforms to the Evolver interface.
/// \tparam T The type to check for conformity to the Evolver inteface.
template <typename T>
static constexpr auto is_updater_v = 
  std::is_base_of<Updater<std::decay_t<T>>, std::decay_t<T>>::value;

}} // namespace fluid::scheme


#endif // FLUIDITY_SCHEME_INTERFACES_UPDATER_HPP
