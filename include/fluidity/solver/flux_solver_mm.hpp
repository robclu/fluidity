//==--- fluidity/solver/flux_solver_mm.hpp ----------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  flux_solver_mm.hpp
/// \brief This file defines a class which combines the reconstruction and flux
///        evaluation and allows the face fluxes for a state to be solved. This
///        implemantation can be used in multi-material simulations.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_FLUX_SOLVER_MM_HPP
#define FLUIDITY_SOLVER_FLUX_SOLVER_MM_HPP

#include <fluidity/dimension/thread_index.hpp>
#include <fluidity/utility/portability.hpp>
#include <fluidity/utility/type_traits.hpp>

namespace fluid  {
namespace solver {

template <typename T, typename ReconMethod, typename FluxMethod>
struct FaceFlux {
  /// Defines the type of the data.
  using value_t         = std::decay_t<T>;
  /// Defines the tyoe of the reconstructor for the solver.
  using reconstructor_t = std::decay_t<ReconMethod>;
  /// Defines the type of the flux evaluator for the solver.
  using flux_method_t   = std::decay_t<FluxMethod>;

  /// Constructor which initializes the flux mehtod computer.
  /// \param[in] dtdh The time-space distrectization.
  fluidity_host_device FaceFluxMM(value_t dtdh) : _dtdh(dtdh) {}

  /// Returns the difference between the face fluxes, and thus returns:
  ///
  /// \begin{equation}
  ///   \delta F = F_{i-1/2} - F{i+1/2}
  /// \end{equation}
  ///
  /// It is thus the same as calling ``fflux.backward() - fflux.forward()```.
  ///
  /// \param[in] it   An iterator which iterates over the state data.
  /// \tparam    It   The type of the iterator over the state data.
  /// \tparam    Dim  The type of the dimension identifier.
  template <typename It, typename Dim>
  fluidity_host_device auto flux_delta(It&& it, Dim dim) const
  {
    return backward(std::forward<It>(it), dim) -
           forward(std::forward<It>(it), dim);
  }

  /// Solves for the face flux between the state pointed to by the iterator and
  /// the state state in the forward direction in the given dimension,
  /// reconstructing the data before using the appropriate face flux.
  /// \param[in] state_it An iterator which iterates over the state data.
  /// \tparam    Iterator The type of the iterator over the state data.
  /// \tparam    Value    The value which defines the dimension to compute the
  ///                     flux in terms of.
  template <typename Iterator, typename Dim>
  fluidity_host_device auto forward(Iterator&& state_it, Dim dim) const
  {
    const auto flux = flux_method_t::get(_material, _dtdh, dim);
    return flux(_recon.forward_left(state_it, _material, _dtdh, dim),
                _recon.forward_right(state_it, _material, _dtdh, dim));
  }

  /// Solves for the face flux between the state pointed to by the iterator and
  /// the state state in the forward direction in the given dimension \p dim,
  /// reconstructing the data before using the appropriate face flux. In
  /// addition to reconstructing the data, the \p predicate is applied to the
  /// left and right reconstructed inputs to the flux solver, before the flux is
  /// sovled for. For example, given a predicate $$p()$$, the following is the
  /// resultant flux:
  ///
  /// \code{cpp}
  /// // recon() =  reconstruction operation
  /// delta_l = recon(left input state from iterator)
  /// delta_r = recon(right input state from iterator)
  ///
  /// l_flux_input, r_flux_input = p(delta_l, delta_r)
  ///
  /// flux = flux_solver(l_flux_input, r_flux_input)
  /// \endcode
  ///
  /// The predicate must have the form ``void p(state_t& l, state_t& r) const``
  /// where the inputs ``l, r`` are references to the state input vectors, 
  /// which are modified by the predicate before being used to solve the flux.
  ///
  /// \param[in] it     An iterator which iterates over the state data.
  /// \param[in] dim    The dimension to solve the flux in terms of.
  /// \param[in] pred   The predicate to apply to the flux input.
  /// \param[in] args   Additional arguments for the predicate.
  /// \tparam    It     The type of the iterator over the state data.
  /// \tparam    Dim    The type of the dimension to solve the flux for.
  /// \tparam    Pred   The type of the predicate to apply to the inputs.
  /// \tparam    Args   The type of additional arguments for the predicate.
  template <typename It, typename Dim, typename Pred, typename... Args>
  fluidity_host_device auto
  forward(It&& it, Dim dim, Pred&& pred, Args&&... args) const
  {
    const auto flux = flux_method_t::get(_material, _dtdh, dim);
    auto left       = _recon.forward_left(it, _material, _dtdh, dim);
    auto right      = _recon.forward_right(it, _material, _dtdh, dim);
    pred(left, right, std::forward<Args>(args)...);
    return flux(left, right);
  }

  /// Solves for the face flux between the state pointed to by the iterator and
  /// the state state in the backward direction in the given dimension,
  /// reconstructing the data before using the appropriate face flux.
  /// \param[in] state_it An iterator which iterates over the state data.
  /// \tparam    Iterator The type of the iterator over the state data.
  /// \tparam    Value    The value which defines the dimension to compute the
  ///                     flux in terms of.
  template <typename Iterator, typename Dim>
  fluidity_host_device auto
  backward(Iterator&& state_it, Dim dim) const
  {
    const auto flux = flux_method_t::get(_material, _dtdh, dim);
      return flux(_recon.backward_left(state_it, _material, _dtdh, dim) ,
                  _recon.backward_right(state_it, _material, _dtdh, dim));
  }

  /// Solves for the face flux between the state pointed to by the iterator and
  /// the state state in the backward direction in the given dimension \p dim,
  /// reconstructing the data before using the appropriate face flux. In
  /// addition to reconstructing the data, the \p predicate is applied to the
  /// left and right reconstructed inputs to the flux solver, before the flux is
  /// sovled for. For example, given a predicate $$p()$$, the following is the
  /// resultant flux:
  ///
  /// \code{cpp}
  /// // recon() =  reconstruction operation
  /// delta_l = recon(left input state from iterator)
  /// delta_r = recon(right input state from iterator)
  ///
  /// l_flux_input, r_flux_input = p(delta_l, delta_r)
  ///
  /// flux = flux_solver(l_flux_input, r_flux_input)
  /// \endcode
  ///
  /// The predicate must have the form ``void p(state_t& l, state_t& r) const``
  /// where the inputs ``l, r`` are references to the state input vectors, 
  /// which are modified by the predicate before being used to solve the flux.
  ///
  /// \param[in] it     An iterator which iterates over the state data.
  /// \param[in] dim    The dimension to solve the flux in terms of.
  /// \param[in] pred   The predicate to apply to the flux input.
  /// \param[in] args   Additional arguments for the predicate.
  /// \tparam    It     The type of the iterator over the state data.
  /// \tparam    Dim    The type of the dimension to solve the flux for.
  /// \tparam    Pred   The type of the predicate to apply to the inputs.
  /// \tparam    Args   The type of additional arguments for the predicate.
  template <typename It, typename Dim, typename Pred, typename... Args>
  fluidity_host_device auto
  backward(It&& it, Dim dim, Pred&& pred, Args&&... args) const
  {
    const auto flux = flux_method_t::get(_material, _dtdh, dim);
    auto left       = _recon.backward_left(it, _material, _dtdh, dim);
    auto right      = _recon.backward_right(it, _material, _dtdh, dim);
    pred(left, right, std::forward<Args>(args)...);
    return flux(left, right);
  } 

 private:
  reconstructor_t _recon;    //!< The reconstructor for the data.
  value_t         _dtdh;     //!< The scaling factor for the computation.
};

}} // namespace fluid::solver

#endif // FLUIDITY_SOLVER_FLUX_SOLVER_MM_HPP