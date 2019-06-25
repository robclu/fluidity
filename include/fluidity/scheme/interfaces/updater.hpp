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
  /// Returns the width required by the evolver. This is the number of cells
  /// on a single side which are required.
  constexpr auto width() const
  {
    return impl()->width();
  }

  /// Implemenation of the function to update the \p it_out data using the
  /// \p it_in data and the evaluator.
  /// \param[in] it_in    The iterable input data to use to evolve.
  /// \param[in] it_out   The iteratable output data to update.
  /// \param[in] dt       The time resolution to use for the update.
  /// \param[in] dh       The spacial resolution to use for the update.
  /// \tparam    ItIn     The type of the input iterator.
  /// \tparam    ItOut    The type of the output iterator.
  /// \tparam    T        The type of the timestep and resolution.
  template <typename ItIn, typename ItOut, typename T>
  fluidity_host_device void
  update(ItIn&& it_in, ItOut&& it_out, T dt, T dh) const
  {
    static_assert(is_multidim_iter_v<ItIn>, 
                  "Input iterator must be a multidimensional iterator!");
    static_assert(is_multidim_iter_v<ItOut>, 
                  "Output iterator must be a multidimensional iterator!");

    return impl()->update_impl(std::forward<ItIn>(it_in)  ,
                               std::forward<ItOut>(it_out),
                               dt                         ,
                               dh                         );
  }

  /// Implemenation of the function to update the \p it_out data using the
  /// \p it_in data and the evaluator.
  ///
  /// \pre The iterators, \p it_in, \p it_out, \p v_it, must be offset to the
  ///      cells which will be used (\p it_in) and set (\p it_out). 
  ///
  /// \param[in] it_in    The iterable input data to use to evolve.
  /// \param[in] it_out   The iteratable output data to update.
  /// \param[in] dt       The time resolution to use for the update.
  /// \param[in] dh       The spacial resolution to use for the update.
  /// \param[in] v_it     Additional iterable data used in the update.
  /// \tparam    ItIn     The type of the input iterator.
  /// \tparam    ItOut    The type of the output iterator.
  /// \tparam    T        The type of the timestep and resolution.
  /// \tparam    VIt      The type of the addition iterable data.
  template <typename ItIn ,
            typename ItOut,
            typename T    ,
            typename VIt  , multiit_enable_t<VIt> = 0>
  fluidity_host_device void
  update(ItIn&& it_in, ItOut&& it_out, T dt, T dh, VIt&& v_it) const
  {
    static_assert(is_multidim_iter_v<ItIn>, 
                  "Input iterator must be a multidimensional iterator!");
    static_assert(is_multidim_iter_v<ItOut>, 
                  "Output iterator must be a multidimensional iterator!");

    return impl()->update_impl(std::forward<ItIn>(it_in)  ,
                               std::forward<ItOut>(it_out),
                               dt                         ,
                               dh                         ,
                               std::forward<VIt>(v_it)    );
  }


  /// Implemenation of the function to update the \p it_out data using the
  /// \p it_in data and the evaluator.
  ///
  /// \todo Add the interface to which the \p f functor must conform.
  ///
  /// \param[in] it_in    The iterable input data to use to evolve.
  /// \param[in] it_out   The iteratable output data to update.
  /// \param[in] dt       The time resolution to use for the update.
  /// \param[in] dh       The spacial resolution to use for the update.
  /// \param[in] f        A functor which is used in the update.
  /// \param[in] args     Additional arguments for the functor.
  /// \tparam    ItIn     The type of the input iterator.
  /// \tparam    ItOut    The type of the output iterator.
  /// \tparam    T        The type of the timestep and resolution.
  /// \tparam    F        The type of the functor.
  /// \tparam    Args     The types of the additional functor arguments.
  template <typename    ItIn ,
            typename    ItOut,
            typename    T    ,
            typename    F    ,
            typename... Args , nonmultiit_enable_t<F> = 0>
  fluidity_host_device void
  update(ItIn&& it_in, ItOut&& it_out, T dt, T dh, F&& f, Args&&... args) const
  {
    static_assert(is_multidim_iter_v<ItIn>, 
                  "Input iterator must be a multidimensional iterator!");
    static_assert(is_multidim_iter_v<ItOut>, 
                  "Output iterator must be a multidimensional iterator!");

    return impl()->update_impl(std::forward<ItIn>(it_in)  ,
                               std::forward<ItOut>(it_out),
                               dt                         ,
                               dh                         ,
                               std::forward<F>(f)         ,
                               std::forward<Args>(args)...);
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
