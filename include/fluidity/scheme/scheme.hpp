//==--- fluidity/scheme/scheme.hpp ------------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  scheme.hpp
/// \brief This file defines an interface for a scheme. A scheme is simply a
///        functor which is used for time evolution, with a specific interface.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_SCHEME_HPP
#define FLUIDITY_SCHEME_SCHEME_HPP

namespace fluid  {
namespace scheme {

/// The Scheme class provides an interface to which numerical schemes must
/// conform.
/// \tparam SchemeImpl The implementation of the scheme interface.
template <typename SchemeImpl>
class Scheme {
  /// Defines the type of the stencil implementation.
  using impl_t = SchemeImpl;

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
  /// Returns the width of the scheme.
  constexpr auto width() const
  {
    return impl()->width();
  }

  /// Overload of the function call operator to invoke the scheme on the data
  /// for the specified \p dim dimension.
  /// \param[in] it     The iterable data to apply the stencil to.
  /// \param[in] h      The delta for the stencil.
  /// \param[in] dim    The dimension to compute the difference in.
  /// \tparam    It     The type of the iterator.
  /// \tparam    T      The type of the delta.
  /// \tparam    Dim    The type of the dimension specifier.
  template <typename It, typename T, typename Dim, typename... Args>
  fluidity_host_device auto operator()(It&& it, T h, Dim dim, Args&&...) const
  {
    static_assert(is_multidim_iter_v<It>, 
                  "Iterator must be a multidimensional iterator!");
    return impl()->invoke(std::forward<It>(it)       ,
                          h                          ,
                          dim                        ,
                          std::forward<Args>(args)...);
  }
};

//==--- Traits -------------------------------------------------------------==//

/// Returns true if the type T conforms to the Scheme interface.
/// \tparam T The type to check for conformity to the Scheme inteface.
template <typename T>
static constexpr auto is_scheme_v = 
  std::is_base_of<Scheme<std::decay_t<T>>, std::decay_t<T>>::value;

}} // namespace fluid::scheme


#endif // FLUIDITY_SCHEME_SCHEME_HPP
