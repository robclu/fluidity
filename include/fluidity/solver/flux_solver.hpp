//==--- fluidity/solver/flux_solver.hpp -------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  flux_solver.hpp
/// \brief This file defines a class which combines the reconstruction and flux
///        evaluation and allows the face fluxes for a state to be solved.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_FLUX_SOLVER_HPP
#define FLUIDITY_SOLVER_FLUX_SOLVER_HPP

#include <fluidity/utility/portability.hpp>
#include <fluidity/utility/type_traits.hpp>

namespace fluid  {
namespace solver {

template <typename ReconMethod, typename FluxMethod, typename Material>
struct FaceFlux {
  /// Defines the tyoe of the reconstructor for the solver.
  using reconstructor_t = std::decay_t<ReconMethod>;
  /// Defines the type of the flux evaluator for the solver.
  using flux_method_t   = std::decay_t<FluxMethod>;
  /// Defines the type of the material.
  using material_t      = std::decay_t<Material>;
  /// Defines the type of the data.
  using value_t         = typename material_t::value_t;

  /// Constructor which initializes the flux mehtod computer.
  /// \param[in] mat  The material in which the face exists.
  /// \param[in] dtdh The time-space distrectization.
  fluidity_host_device FaceFlux(material_t mat, value_t dtdh) 
  : _material(mat), _dtdh(dtdh) {}

  /// Returns the difference between the face fluxes, and thus returns:
  /// \begin{equation}
  ///   \delta F = F_{i-1/2} - F{i+1/2}
  /// \end{equation}
  /// It is thus the same as calling ``fflux.backward() - fflux.forward()```.
  /// \param[in] state_it An iterator which iterates over the state data.
  /// \tparam    Iterator The type of the iterator over the state data.
  /// \tparam    Value    The value which defines the dimension to compute the
  ///                     flux difference in terms of.
  template <typename Iterator, std::size_t Value>
  fluidity_host_device auto
  flux_delta(Iterator&& state_it, Dimension<Value>) const
  {
    constexpr auto dim = Dimension<Value>{};
    return backward(std::forward<Iterator>(state_it), dim) -
           forward(std::forward<Iterator>(state_it), dim);
  }

  /// Solves for the face flux between the state pointed to by the iterator and
  /// the state state in the forward direction in the given dimension,
  /// reconstructing the data before using the appropriate face flux.
  /// \param[in] state_it An iterator which iterates over the state data.
  /// \tparam    Iterator The type of the iterator over the state data.
  /// \tparam    Value    The value which defines the dimension to compute the
  ///                     flux in terms of.
  template <typename Iterator, std::size_t Value>
  fluidity_host_device auto forward(Iterator&& state_it, Dimension<Value>) const
  {
    constexpr auto dim = Dimension<Value>();
    const auto flux    = flux_method_t::get(_material, _dtdh, dim);
    return flux(_recon.forward_left(state_it, _material, _dtdh, dim),
                _recon.forward_right(state_it, _material, _dtdh, dim));
  }

  /// Solves for the face flux between the state pointed to by the iterator and
  /// the state state in the backward direction in the given dimension,
  /// reconstructing the data before using the appropriate face flux.
  /// \param[in] state_it An iterator which iterates over the state data.
  /// \tparam    Iterator The type of the iterator over the state data.
  /// \tparam    Value    The value which defines the dimension to compute the
  ///                     flux in terms of.
  template <typename Iterator, std::size_t Value>
  fluidity_host_device auto
  backward(Iterator&& state_it, Dimension<Value>) const
  {
    constexpr auto dim = Dimension<Value>();
    const auto flux    = flux_method_t::get(_material, _dtdh, dim);
    return flux(_recon.backward_left(state_it, _material, _dtdh, dim) ,
                _recon.backward_right(state_it, _material, _dtdh, dim));
  }

 private:
  reconstructor_t _recon;    //!< The reconstructor for the data.
  material_t      _material; //!< The material to copute the flux for.
  value_t         _dtdh;     //!< The scaling factor for the computation.
};

}} // namespace fluid::solver

#endif // FLUIDITY_SOLVER_FLUX_SOLVER_HPP