//==--- fluidity/scheme/evaluators/levelset_hailtonian.hpp - -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  levelset_hamiltonian.hpp
/// \brief This file defines an implementation of the Evaluatable interface
///        which evaluates data based on the levelset Hamiltonian.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_EVALUATORS_LEVELSET_HAMILTONIAN_HPP
#define FLUIDITY_SCHEME_EVALUATORS_LEVELSET_HAMILTONIAN_HPP

#include "../interfaces/evaluatable.hpp"
#include "../schemes/godunov_upwind.hpp"

namespace fluid     {
namespace scheme    {
namespace evaluator {

/// The LevelsetHamiltonian class implements an evaluatable class which
/// can evaluate the levelset hamiltonian using a Godunov upwind spatial
/// discretization and the strncil provided by the template type. This allows
/// any type of stencil for the derivatives to be used to compose numerous
/// levelset Hamiltonians with different accuracy.
/// \tparam Stencil The spatial stencil to use with the Godunov upwind method.
template <typename Stencil>
class LevelsetHamiltonian : public Evaluatable<LevelsetHamiltonian<Stencil>> {
 private:
  /// Defines the type of the scheme used to evaluate the Hamiltonian.
  using scheme_t = std::decay_t<GodunovUpwind<Stencil>>;

 public:
  /// Returns the width required when evaluating. The width is the maximum
  /// offset (in any dimension) from a cell to another cell whose data needs to
  /// be used by the cell when performing the evaluation.
  constexpr auto width() const -> std::size_t
  {
    return scheme_t{}.width();
  }

  /// This method evaluates the data, and returns the computed value. This
  /// overload is only enabled when the \p v_it is a multi-dimensional iterator.
  ///
  /// \pre The \p it input data iterator, and \p v_it velocity iterators are
  ///      assumed to already be offset to the point to the cells which should
  ///      be used to perform the evaluation. This therefore assumes that the
  ///      iterators have been offset.
  /// 
  /// \param[in] it   The iterable data to evaluate.
  /// \param[in] dh   The discretization which the evaluation can use.
  /// \param[in] v_it Iterable data to multiply with the iterable data.
  /// \param[in] args Additional arguments for the evaluation.
  /// \tparam    It   The type of the data iterator. This must be a multi
  ///                 dimensional iterator.
  /// \tparam    T    The type of the discretization value. It must be possible
  ///                 to perform mathematical operations with this type and the
  ///                 data iterated over.
  /// \tparam    VIt  The tyoe of the multiplication data. This must be a multi
  ///                 dimensional iterator.
  /// \tparam    Args The types of the additional arguments.
  template <typename    It  ,
            typename    T   ,
            typename    VIt ,
            typename... Args, multiit_enable_t<VIt> = 0>
  fluidity_host_device auto
  evaluate(It&& it, T dh, VIt&& v_it, Args&&... args) const
  {
    static_assert(is_multidim_iter_v<It>,
      "Iterator for Evaluatable must be a multi-dimensional iterator!");
    
    using value_t = std::decay_t<decltype(*v_it)>;
    return *v_it * (*v_it <= value_t{0}
      ? scheme_t().backward(std::forward<It>(it)       ,
                            dh                         ,
                            std::forward<Args>(args)...)
      : scheme_t().forward(std::forward<It>(it)       ,
                           dh                         ,
                           std::forward<Args>(args)...));
  }

  /// This method evaluates the data, and returns the computed value. This
  /// overload is only enabled when the \p f is not a multi-dimensional
  /// iterator.
  /// 
  /// \pre The \p it input data iterator, and \p v_it velocity iterators are
  ///      assumed to already be offset to the point to the cells which should
  ///      be used to perform the evaluation. This therefore assumes that the
  ///      iterators have been offset.
  ///
  /// \param[in] it   The iterable data to evaluate.
  /// \param[in] dh   The discretization which the evaluation can use.
  /// \param[in] f    A functor to apply to the data.
  /// \param[in] args Additional arguments for the evaluation.
  /// \tparam    It   The type of the data iterator. This must be a multi
  ///                 dimensional iterator.
  /// \tparam    T    The type of the discretization value. It must be possible
  ///                 to perform mathematical operations with this type and the
  ///                 data iterated over.
  /// \tparam    F    The type of the functor.
  /// \tparam    Args The types of the additional arguments.
  template <typename    It  ,
            typename    T   ,
            typename    F   ,
            typename... Args, nonmultiit_enable_t<F> = 0>
  fluidity_host_device auto
  evaluate(It&& it, T dh, F&& f, Args&&... args) const
  {
    static_assert(is_multidim_iter_v<It>,
      "Iterator for Evaluatable must be a multi-dimensional iterator!");

    const auto f_val = f(it, std::forward<Args>(args)...);
    using value_t    = std::decay_t<decltype(f_val)>;
    return f_val * (*f_val <= value_t{0}
      ? scheme_t().backward(std::forward<It>(it)       ,
                            dh                         ,
                            std::forward<Args>(args)...)
      : scheme_t().forward(std::forward<It>(it)       ,
                           dh                         ,
                           std::forward<Args>(args)...));
  }
};

}}} // namespace fluid::scheme::evaluator


#endif // FLUIDITY_SCHEME_EVALUATORS_LEVELSET_HAMILTONIAN_HPP

