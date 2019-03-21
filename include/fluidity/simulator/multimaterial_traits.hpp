//==--- fluidity/simulator/multimaterial_traits.hpp -------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  multimaterial_traits.hpp
/// \brief This file provides traits for the multimaterial simulator 
///        implementation.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SIMULATOR_MULTIMATERIAL_TRAITS_HPP
#define FLUIDITY_SIMULATOR_MULTIMATERIAL_TRAITS_HPP

#include "multimaterial_simulation_data.hpp"
#include <fluidity/container/tuple.hpp>
#include <fluidity/levelset/levelset.hpp>
#include <fluidity/material/material.hpp>
#include <fluidity/setting/data_option.hpp>
#include <fluidity/setting/dimension_option.hpp>
#include <fluidity/setting/execution_option.hpp>
#include <fluidity/setting/flux_method_option.hpp>
#include <fluidity/setting/limit_form_option.hpp>
#include <fluidity/setting/limiter_option.hpp>
#include <fluidity/setting/material_option.hpp>
#include <fluidity/setting/option_manager.hpp>
#include <fluidity/setting/reconstruction_option.hpp>
#include <fluidity/setting/solve_option.hpp>
#include <fluidity/setting/parameter/parameter_managers.hpp>
#include <fluidity/solver/flux_solver.hpp>
#include <fluidity/solver/material_loader.hpp>
#include <fluidity/state/state.hpp>
#include <fluidity/utility/type_traits.hpp>

namespace fluid {
namespace sim   {

namespace detail {

template <typename SimTraits, typename Materials, typename Levelset>
struct MaterialTraits {
  /// Defines the type of the container which stores the materials.
  using container_t = Tuple<material::Material<Materials, Levelset>>;

  /// Defines the number of materials.
  static constexpr auto num_materials      = std::size_t{1};
  /// Defines that this is not multi-material.
  static constexpr auto is_multimaterial_v = false;
};

template <typename SimTraits, typename... Es, typename Levelset>
struct MaterialTraits<SimTraits, Tuple<Es...>, Levelset> {
  /// Defines the type of the levelset.
  using levelset_t   = Levelset;
  /// Defines the traits for the simulation.
  using sim_traits_t = SimTraits;

  /// Defines the type of a single material with a levelset.
  /// \tparam E The equation of state for the material.
  template <typename E>
  using material_t = material::Material<E, levelset_t>;

  /// Defines the multimaterial simulation data struct given the equation of
  /// state.
  /// \tparam E the equations of state for the material.
  using mm_sim_data_t = Tuple<
    MultimaterialSimData<
      sim_traits_t, material_t<Es>, sim_traits_t::exec_t::device>...>;

  /// Defines the number of materials.
  static constexpr auto num_materials = sizeof...(Es);
};

} // namespace detail

template <typename SimTraits, typename Es, typename Levelset>
using material_traits_t = detail::MaterialTraits<SimTraits, Es, Levelset>;

/// Defines traits for a simulator implementation which can simulate multiple
/// materials in the domain.
/// \tparam SimBase The type of the base simulator class which can be created.
/// \tparam SimImpl The type of the simulator implementation.
/// \tparam Ts      The traits for the simulator implementation.
template <typename SimBase, typename SimImpl, typename... Ts>
struct MultimaterialSimTraits {
  /// Defines the default data type to use.
  using def_data_t    = double;
  /// Defines the default number of dimensions.
  using def_dims_t    = Num<1>;
  /// Defines the default material to use.
  //using def_mat_t     = material::IdealGas<def_data_t>;
  using def_mat_t     = Tuple<material::IdealGas<def_data_t>,
                              material::IdealGas<def_data_t>>;
  /// Defines the default limiting form.
  using def_form_t    = limit::cons_form_t;
  /// Defines the default limiter to use.
  using def_limiter_t = limit::Superbee<def_form_t>;
  /// Defines the default reconstructor to use.
  using def_recon_t   = recon::MHUReconstructor<def_limiter_t>;
  /// Defines the default flux method to use.
  using def_flux_t    = flux::Force;
  // Defines the default execution type for the simulation.
  using def_exec_t    = fluid::exec::gpu_type;

  /// Defines the type of the state data to store, always conservative.
  using data_t     = type_at_t<0, def_data_t, Ts...>;
  /// Defines the number of dimensions for the simulation.
  using dim_t      = type_at_t<1, def_dims_t, Ts...>;
  /// Defines the type of the material for the simulation.
  using materials_t = type_at_t<2, def_mat_t, Ts...>;
  /// Defines the form of limiting.
  using lform_t    = type_at_t<3, def_form_t, Ts...>;
  /// Defines the type of the limiter to use.
  using limiter_t  = type_at_t<4, def_limiter_t, Ts...>;
  /// Defines the type of the reconstruction method to use.
  using recon_t    = type_at_t<5, def_recon_t, Ts...>;
  /// Defines the type of the flux method to use for solving.
  using flux_t     = type_at_t<6, def_flux_t, Ts...>;
  /// Defines execution policy for the simulator.
  using exec_t     = type_at_t<7, def_exec_t, Ts...>;

  /// Defines the type of the level sets used.
  using levelset_t = levelset::LevelSet<data_t, dim_t::value, exec_t::device>; 

  /// Defines the materials for the simulation.
  //using mat_traits_t = material_traits_t<materials_t, levelset_t>;

  /// TODO: Change this to be per material, or change the solver to take the
  ///       material is a template parameter.

  /// Defines the type of the data loader.
  using loader_t     = solver::BoundaryLoader<recon_t::width>;
  /// Defines the type of the face flux solver.
  using face_flux_t  = solver::FaceFlux<recon_t, flux_t, materials_t>;
  /// Defines the default type of solver.
  using def_solver_t = solver::SplitSolver<face_flux_t, loader_t, dim_t>;
  /// Defines the type of the solver.
  using solver_t     = type_at_t<8, def_solver_t, Ts...>;

  /// Defines the type of the loader for the material ghost state data.
  using mm_loader_t  = solver::MaterialLoader;

  /// Defines the type of the multimaterial solver for a material Mat.
  /// \tparam EOS The equation of state for the material for which the solver 
  /// must solve.
  template <typename Eos>
  using mm_solver_t = 
    solver::SplitSolver<
      solver::FaceFlux<recon_t, flux_t, Eos>, loader_t, dim_t>;

  /// Defines the storage format for the state data.
  static constexpr auto state_layout = StorageFormat::row_major;

  /// Defines the type for primitive states for this solver.
  using primitive_t = 
    state::primitive_t<data_t, dim_t::value, 0, state_layout>;

  /// Defines the type for conservative states for this solver.
  using conservative_t =
    state::conservative_t<data_t, dim_t::value, 0, state_layout>;

  /// Defines the type of the states for this solver.
  using state_t = conservative_t;

  /// Defines the type of the option manger to configure the simulation traits.
  using option_manager_t =
    setting::OptionManager<
      SimBase                                            ,
      SimImpl                                            ,
      setting::param_manager_t                           ,
      setting::DataOption                                ,
      setting::DimensionOption                           ,
      setting::MaterialOption<data_t>                    ,
      setting::LimitFormOption                           ,
      setting::LimiterOption<lform_t>                    ,
      setting::ReconOption<limiter_t>                    ,
      setting::FluxMethodOption                          ,
      setting::ExecutionOption                           ,
      setting::SolverOption<face_flux_t, loader_t, dim_t>>;

  /// Defines the number of dimensions for the simulation.
  static constexpr auto dimensions = std::size_t{dim_t::value};
};

}} // namespace fluid::sim

#endif // FLUIDITY_SIMULATOR_MULTIMATERIAL_TRAITS_HPP