//==--- fluidity/simulator/multimaterial_simulator.hpp ----- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  multimaterial_simulator.hpp
/// \brief This file defines a simulator which can solve multimaterials.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SIMULATOR_MULTIMATERIAL_SIMULATOR_HPP
#define FLUIDITY_SIMULATOR_MULTIMATERIAL_SIMULATOR_HPP

#include "parameters.hpp"
#include "simulator.hpp"
#include "simulation_data.hpp"
#include "multimaterial_traits.hpp"
#include "wavespeed_initialization.hpp"
#include <fluidity/algorithm/fill.hpp>
#include <fluidity/algorithm/max_element.hpp>
#include <fluidity/dimension/dimension_info.hpp>
#include <fluidity/ghost_fluid/load_ghost_cells.hpp>
#include <fluidity/ghost_fluid/simple_ghost_fluid.hpp>
#include <fluidity/levelset/first_order_evolution.hpp>
#include <fluidity/levelset/levelset_evolution.hpp>
#include <fluidity/levelset/levelset_projection.hpp>
#include <fluidity/levelset/velocity_initialization.hpp>
#include <fluidity/material/combine_materials.hpp>
#include <fluidity/setting/parameter/configurable.hpp>
#include <fluidity/utility/timer.hpp>
#include <fluidity/utility/type_traits.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <experimental/filesystem>

namespace fluid {
namespace sim   {

namespace fs = std::experimental::filesystem;

/// The MultimaterialSimulator class implements the simulation interface to
/// simulate the case where there are multiple materials in the domain.
/// \tparam Traits The traits which define the simulation paramters.
template <typename... Traits>
class MultimaterialSimulator final : 
  public Simulator,
  public setting::Configurable<MultimaterialSimulator<Traits...>> {
 public:
  /// Defines the type of this simulator.
  using this_t               = MultimaterialSimulator;
  /// Defines the type of the base simulator class.
  using base_t               = Simulator;
  /// Defines the type of the filler container .
  using fillinfo_container_t = typename base_t::fillinfo_container_t;

 private:
  /// Defines the type of the traits for the simulation implementation.
  using traits_t     = MultimaterialSimTraits<base_t, this_t, Traits...>;

  /// Defines the type of the state data to store, always conservative.
  using state_t      = typename traits_t::state_t;
  /// Defines the data type used in the state vector.
  using value_t      = typename state_t::value_t;
  /// Defines the type of the material for the simulation.
  using materials_t  = typename traits_t::materials_t;
  /// Defines the type of the reconstruction method to use.
  using recon_t      = typename traits_t::recon_t;
  /// Defines execution policy for the simulator.
  using exec_t       = typename traits_t::exec_t;
  /// Defines the type of the solver.
  using solver_t     = typename traits_t::solver_t;
  /// Defines the type of the levelset used to define materials.
  using levelset_t   = typename traits_t::levelset_t;
  /// Defines the type of the loader of the ghost material state data.
  using mm_loader_t  = typename traits_t::mm_loader_t;

  /// Defines the type of the parameter container.
  using params_t    = Parameters<value_t>;
  /// Defines the type of the boundary setter.
  using setter_t    = solver::BoundarySetter;

  /// Defines the type of the data storage for the simulation.
  //using storage_t   = SimulationData<traits_t, exec_t::device>;

  /// Defines the storage type for each of the materials.
  //using storage_t   = multimaterial_data_t<traits_t>;

  /// Defines the traits for the materials.
  using mat_traits_t  = material_traits_t<traits_t, materials_t, levelset_t>;
  /// Defines the type which holds the simulation data.
  using mm_sim_data_t = typename mat_traits_t::mm_sim_data_t;

  /// Defines the type of the ghost fluid loader.
  using ghost_loader_t     = ghost::SimpleGFM<3>;
  /// Defines the type of the levelset solver.
  using levelset_evolver_t = typename traits_t::ls_evolver_t;

  /// Defines a constexpr instance of the execution polcity.
  static constexpr auto execution_policy = exec_t{};
  /// Defines the number of materials in the simulation.
  static constexpr auto num_materials    = mat_traits_t::num_materials;
 public:
  /// Defines the type of the option manager which is used to configure the
  /// types to use for the simulation.
  using option_manager_t = typename traits_t::option_manager_t;

  /// Defines the number of spacial dimensions in the simulation.
  static constexpr auto dimensions = state_t::dimensions;
  /// Defines the amount of padding used for the simulation.
  static constexpr auto padding    = solver_t::loader_t::padding;

  /// Creates the simulator.
  MultimaterialSimulator() : _params{dimensions} {}

  /// Cleans up all resources acquired by the simulator.
  ~MultimaterialSimulator() {}

  /// Runs the simulation until completion.
  void simulate() override;

  /// Fills the simulator with simulation data for a simulator, where each of
  /// the fillers store the name of the property which they are filling, and 
  /// a function object which can fill values based on their position in the
  /// space.
  /// \param[in] fillers A container of fillers for filling the data.
  void fill_data(fillinfo_container_t&& fillers) override;

  /// Configures the simulator to use the specified CFL number.
  /// \param[in] cfl The CFL number to use for the simulation.
  void configure_cfl(double cfl) override;

  /// Configures the simulator to set size and resolution of a dimension \p dim.
  /// \param[in] dim   The dimension to specify.
  /// \param[in] start The start value of the dimension.
  /// \param[in] end   The end value of the dimension.
  void configure_dimension(std::size_t dim, double start, double end) override
  {
    _params.domain.set_dimension(dim, start, end);
    for_each(_mm_data, [&] (auto& mm_data)
    {
      mm_data.resize_dim(dim, _params.domain.elements(dim));
    });
  }

  /// Configures the simulator to simulate for a maximum number of iterations.
  /// \param[in] iters  The maximum number of iterations to simulate for.
  void configure_max_iterations(std::size_t iters) override;  

  /// Configures the simulator to use the \p resolution for the domain.
  /// \param[in] resolution The resolution to use for the domain.
  void configure_resolution(double resolution) override;

  /// Configures the simulator to simulate until a certain simulation time.
  /// \param[in] sim_time The time to run the simulation until.
  void configure_sim_time(double sim_time) override;

  /// Prints the results of the simulation to the standard output stream.
  void print_results() const override;

  /// Writes the results of the simulation to the \p path using the \p prefix
  /// appended to the property which is output. If \p path = "", then the
  /// current working directory is used as the path.
  /// \param[in] file_path   The path (inluding the file prefix) to write to.
  void write_results(std::string file_path) const override;

  /// Writes the results of the simulation to the \p path using the \p prefix
  /// appended to the property which is output. This outputs a separate file for
  /// each of the components of the state (i.e, density, pressure, etc), and
  /// outputs the data in the same format as the domain.
  /// \param[in] path   The file path (including the prefix) to write to.
  void write_results_separate_raw(std::string path) const override;

  /// Adds a material to the simulator, with a specific equation of state, an
  /// initializer for the levelset, and the values of the state data inside the
  /// material. The following in an example usage:
  /// \code{cpp}
  /// simulator->add_material(
  ///   IdealGas<double>{1.4},
  ///   [&] fluidity_host_device (auto it, auto& cell_ids)
  ///   {
  ///     *it = cell_ids[0] < 0.1 ? 0 : 1;
  ///   },
  ///   0.1_rho, 0.5_p, 0.25_v_x
  /// );
  /// \endcode
  ///
  /// The predicate for setting the levelset must take two arguments. The first
  /// is the iterator to the levelset data which must be set, and the second is
  /// a reference to the indices of the cell in each dimension.
  template <typename Eos, typename LSPred, typename... Elements>
  void add_material(Eos&& eos, LSPred&& pred, Elements&&... elements)
  {
    //auto es = make_tuple(std::forward<Elements>(elements)...);
    //for_each(es, [] (auto& e)
    //{
    //  std::cout << e.name() << ": " << e.value << "\n";
    //});

    auto eos_type_set = false;
    for_each(_mm_data, [&] (auto& mm_data)
    {
      using eos_t    = std::decay_t<Eos>;
      using mm_eos_t = std::decay_t<decltype(mm_data.material().eos())>;

      auto& material     = mm_data.material();
      auto& material_out = mm_data.material_out();

      // If the material equation of state type is the same as the one which has
      // been given, and if we have not set a material of this type before and
      // the levelset for the material has not yet been set, then the data can
      // be initialized.
      if (std::is_same<eos_t, mm_eos_t>::value && 
          !eos_type_set && !material.is_initialized())
      {
        material.init_levelset(pred);
      
        // Now set the state data for the material.
        mm_data.set_state_data(std::forward<Elements>(elements)...);
        eos_type_set = true;
      }
    });
  }

  /// Performs the simulation for the multimaterial simulator.
  void simulate_mm()
  {
    // Initialize everything which doesn't depend on the material type.
    _params.print_static_summary();
    auto       cfl   = _params.cfl;
    auto       timer = util::default_timer_t();
    const auto dh    = _params.domain.resolution();

    using host_vel_t = HostTensor<value_t, dimensions>;
    using dev_vel_t  = DeviceTensor<value_t, dimensions>;
    using vel_data_t =
      std::conditional_t<
        std::is_same<exec_t, exec::cpu_type>::value, host_vel_t, dev_vel_t>;

    // Define the velocities for evolving the levelset
    // and resize them to the appropriate size.
    vel_data_t velocities;
    velocities.resize(get<0>(_mm_data).states().total_size());
    
    init_material_data();

    while (_params.continue_simulation())
    {
      //==--- [D] Print iter and levelset data -----------------------------==//
      print_new_iter();
      print_subprocess("Material levelsets");
      for_each(_mm_data, [] (auto& mm_data)
      {
        mm_data.material().levelset().print();
      });
      
      //==-- [R] Loading the ghost cells -----------------------------------==//
      print_subprocess("Loading ghost cells");
      ghost::load_ghost_cells(ghost_loader_t(), _mm_data, dh);

      // [D] print the data for the materials to check correct initialization.
      for_each(_mm_data, [&] (auto& mm_data)
      {
        print_mm_data(mm_data.states(), mm_data.material().eos());
      });

      //==-- [R] Update the timestep for the iteration ---------------------==//
      update_time_delta();

      //==-- [R] Evolve the material data ----------------------------------==//
      evolve_materials();

      //==-- [R] Evolve the levelset data ----------------------------------==//
      evolve_levelsets(velocities);

/*
      print_subprocess("Settings levelset velocities");
      auto v_iter = velocities.multi_iterator();
      levelset::set_velocities(_mm_data, v_iter);

      auto host_vel = velocities.as_host();
      for (const auto& v : host_vel)
      {
          std::cout 
            << std::right
            << std::setfill(' ')
            << std::setw(8)
            << std::setprecision(4)
            << v;
      }
      std::cout << "\n";

      print_subprocess("Evolving levelsets");
      int material_index = 0;
      // Update each of the material levelsets ...
      auto ls_evolver = levelset_evolver_t();
      for_each(_mm_data, [&] (auto& mm_data)
      {
//        print_mm_data(mm_data.states());
        print_subprocess("Evolving levelset", material_index);
        mm_data.material().levelset().print();
        print_mm_data(mm_data.states(), mm_data.material().eos());

        auto ls_it_in  = mm_data.material().levelset().multi_iterator();
        auto ls_it_out = mm_data.material_out().levelset().multi_iterator();
        evolve_levelset(ls_evolver, ls_it_in, ls_it_out, v_iter, _params.dt());

        print_subprocess("Evolved levelset", material_index++);
        mm_data.material_out().levelset().print();

        mm_data.swap_material_levelsets(ls_it_out, ls_it_in);
      });

      // Reinitialize the levelsets ...
      print_subprocess("End if iter data");


      for_each(_mm_data, [&] (auto& mm_data)
      {
        //auto& mat = mm_sim_data.material().levelset();
        // reinitialize_levelset(levelset);

        //print_mm_data(mm_data.states(), mm_data.material().eos());
      });
*/
      _params.update_simulation_info();
    }

    printf("Simulation time : %8lu ms\n", timer.elapsed_time());
    _params.print_final_summary();

    // Combine the material data into the first material
    // so it can be displayed;
    material::combine_into_first(_mm_data);
        
    // Sync the data for the first material.
    get_front(_mm_data).sync_device_to_host();

    // Finalise the data by taking the appropriate states which are inside an
    // appropriate levelset and writing that state to a global array.
    //_data.sync_device_to_host();
  }

 private:
  /// Defines a constexpr instance of a tag which is std::true_type of the batch
  /// size must be fetched for 1 spacial dimension.
  static constexpr auto batch_size_tag = 
    std::integral_constant<bool, traits_t::dimensions == 1>{};

  mm_sim_data_t _mm_data;           //!< Materials for the simulation.
  params_t      _params;            //!< The parameters for the simulation.
  setter_t      _boundary_setter;   //!< The boundary setter.

/*
  void configure(const setting::Parameter* param)
  {
    std::cout << "Don't know how to set this param : " << param->type() << "\n";
  }
*/

  /// Returns the dimension information for the simulator.
  auto dimension_info() const
  {
    auto dim_info = DimInfo<dimensions>();
    unrolled_for<dimensions>([&] (auto i)
    {
      dim_info[i] = get<0>(_mm_data).states().size(i);
      //dim_info[i] = _data.states().size(i);
    });
    return dim_info;
  }

  /// Initializes the material data, ensuring that it is on both the host and
  /// the device.
  void init_material_data()
  {
    print_subprocess("Initializing material data");
    for_each(_mm_data, [&] (auto& mm_data)
    {
      mm_data.initialize();
      mm_data.sync_device_to_host();

      // [D] print the data for the materials to check correct initialization.
      print_mm_data(mm_data.states(), mm_data.material().eos());
    });
  }

  /// Updates the time delta for the simulation by finding the maximum wavespeed
  /// in the materials.
  void update_time_delta()
  {
    print_subprocess("Computing wavespeeds");
    value_t max_ws = 0.0;
      
    // Compute the max wavespeed to use for the time delta ...
    for_each(_mm_data, [&] (auto& mm_data)
    {
      auto& mat = mm_data.material();

      // Get input, output, and wavespeed iterators for this material ...
      auto in  = mm_data.input_iterator();
      auto out = mm_data.output_iterator();
      auto ws  = mm_data.wavespeed_iterator();

      set_wavespeeds(in, ws, mat.eos());
      auto max_speed = max_element(mm_data.wavespeeds().begin(),
                                   mm_data.wavespeeds().end()  );

      // Check if we have found a new best wavespeed, if so, set it.
      if (max_speed > max_ws) { max_ws = max_speed; }  
    });
    _params.update_time_delta(max_ws);
  }

  /// Evolves the data for each of the materials in the simulation.
  void evolve_materials()
  {
    print_subprocess("Evolving materials");
    for_each(_mm_data, [&] (auto& mm_data)
    {
      mm_data.sync_device_to_host();
      print_mm_data(mm_data.states(), mm_data.material().eos());

      mm_data.evolve(_params.dt_dh(), _boundary_setter);
      mm_data.swap_material_state_in_out_data();

      //==-- [D] [Print the updated material] ------------------------------==//
      mm_data.sync_device_to_host();
      print_mm_data(mm_data.states(), mm_data.material().eos());
    });
  }

  /// Evolves the levelset data for the materials. This comprises of a number of
  /// steps, which are:
  ///
  /// - Setting the velocity data for the evolution.
  /// - Updating the levelset data using the velocties.
  ///
  /// After doing the above, the levelsets may overlap or have separated
  /// (creating a vaccuum), so the next step is:
  ///
  /// - Performing a fixup so that holes and overlaps are removed.
  /// 
  /// Finally, the updated and fixed-up levelsets may not be signed distance
  /// funcsions (i.e $| \div \phi | = 0 | may not hold), and thus they need to
  /// be reinitialized, so the final step is:
  ///
  /// - Perform levelset re-initialization.
  ///
  /// \param[in] velocities The velocities to evolve the levelsets with.
  /// \tparam    Velocities The types of the velocities.
  template <typename Velocities>
  void evolve_levelsets(Velocities&& velocities)
  {
    //==-- [D] [Print levelsets] -------------------------------------------==//
    print_subprocess("Levelsets before");
    for_each(_mm_data, [] (auto& mm_data)
    {
      mm_data.material().levelset().print();
    });

    //==-- [D] [Set velocity data] -----------------------------------------==//
    print_subprocess("Settings levelset velocities");
    auto v_iter = velocities.multi_iterator();
    levelset::set_velocities(_mm_data, v_iter);

    //==-- [D] [Print velocity data] ---------------------------------------==//
    auto host_vel = velocities.as_host();
    for (const auto& v : host_vel)
    {
      std::cout << std::right   << std::setfill(' ')
                << std::setw(8) << std::setprecision(4) 
                << v;
    }
    std::cout << "\n";

    //==-- [R] [Perform the evolution] -------------------------------------==//
    evolve_levelsets_impl(v_iter);

    // [D]
    for_each(_mm_data, [] (auto& mm_data)
    {
      mm_data.material().levelset().print();
    });

    //==-- [R] [Perform the projection fixup] ------------------------------==//
    project_levelsets();

    // [D]
    for_each(_mm_data, [] (auto& mm_data)
    {
      mm_data.material().levelset().print();
    });

    //==-- [R] [Re-initialize the levelsets] -------------------------------==//
    //reinit_levelsets(v_iter());
  }

  /// Sets the velocities for the levelset using the velocity and levelset data
  /// from the materials.
  /// \param[in] velocities The velocities to set the data for.
  /// \tparam    Velocities The types of the velocities.
  template <typename Velocities>
  void set_levelset_velocities_from_materials(Velocities&& velocities)
  {
    print_subprocess("Settings levelset velocities");
    levelset::set_velocities(_mm_data, velocities.multi_iterator());

    auto host_vel = velocities.as_host();
    for (const auto& v : host_vel)
    {
      std::cout 
        << std::right
        << std::setfill(' ')
        << std::setw(8)
        << std::setprecision(4)
        << v;
    }
    std::cout << "\n";
  }

  /// Implementation of levelset evolution for all levelsets.
  /// \param[in] v_iter           The iterator over the velocity data.
  /// \tparam    VelocityIterator The type of the velocity iterator.
  template <typename VelocityIterator>
  void evolve_levelsets_impl(VelocityIterator&& v_iter)
  {
    print_subprocess("Evolving levelsets");
      
    // Perform the evolution.
    auto ls_evolver = levelset_evolver_t();
    for_each(_mm_data, [&] (auto& mm_data)
    {
      auto ls_it_in  = mm_data.material().levelset().multi_iterator();
      auto ls_it_out = mm_data.material_out().levelset().multi_iterator();

      fluid::scheme::evolve(levelset_evolver_t(),
                            ls_it_in            ,
                            ls_it_out           ,
                            _params.dt()        ,
                            _params.dh()        ,
                            v_iter              );

      mm_data.swap_material_levelset_in_out_data();
    });
  }

  /// Performs a projection method on the levelsets to ensure that they all
  /// agree on the lolocations of the interfaces and that there are no overlaps
  /// or voids.
  void project_levelsets()
  {
    print_subprocess("Projection levelsets");

    levelset::project(unpack(_mm_data, [] (auto&&... material_data)
    {
      return make_tuple(material_data.material().levelset().multi_iterator()...);
    }));
  }

  //==--- [S - Debug utilities] --------------------------------------------==//
  
  /// Prints a banner for a new iteration for a simulation.
  void print_new_iter() const
  {
    printf("============================================================\n");
    printf("| [S] [New iteration]                                      |\n");
    printf("============================================================\n");
  }

  
  /// Prints that a new subprocess with \p name is starting.  
  /// \param[in] name The name of the new process.
  template <typename Name>
  void print_subprocess(Name name) const
  {
    printf("| [S] [New process] : [ %s ]\n", name);
  }

  /// Prints the state data in the \p states vector from a material with the
  /// equation of state \p eos.
  /// \param[in] states The states to print.
  /// \param[in] eos    The equation of state for the material the state is in.
  template <typename States, typename Eos>
  void print_mm_data(States&& states, Eos&& eos) const
  {
    // Density:
    std::cout << "[D]" << " : ";
    for (const auto& state : states)
    {
      std::cout << std::right      << std::setfill(' ')
                << std::setw(8)    << std::setprecision(4)
                << state.density() << " ";
    }
    std::cout << "\n";

    // Pressure:
    std::cout << "[P]" << " : ";
    for (const auto& state : states)
    {
      std::cout << std::right          << std::setfill(' ')
                << std::setw(8)        << std::setprecision(4)
                << state.pressure(eos) << " ";
    }
    std::cout << "\n";

    // Velocity:
    std::cout << "[V]" << " : ";
    for (const auto& state : states)
    {          
      std::cout << std::right                     << std::setfill(' ')
                << std::setw(8)                   << std::setprecision(4)
                << state.velocity(std::size_t(0)) << " ";
    }
    std::cout << "\n";
  }


  //==--- [E - Debug utilities] --------------------------------------------==//

  /// Implementation of the outputting functionality. The \p stream parameter
  /// is used to determine if the output is written to a file or if it is
  /// sent to the standard output stream.
  /// 
  /// This implementation will print the data in columns, with a column for the
  /// position of the cell in each dimension, and then columns for each element
  /// in the state vector.
  /// 
  /// \param[in] stream   The stream to output the results to.
  /// \param[in] mat_data The material data to output.
  /// \tparam    Stream   The type of the output stream.
  /// \tparam    MatData  The type of the material data to write.
  template <typename Stream, typename MatData>
  void stream_output_ascii(Stream&& stream, const MatData& mat_data) const;

  template <typename Iter, typename Streams>
  void write_blob(Iter iter, Streams& streams, dimx_t) const;

  template <typename Iter>
  void write_blob(Iter iter, std::string prefix, dimx_t) const;

  template <typename Iter>
  void write_blob(Iter iter, std::string prefix, dimy_t) const;

  template <typename Iter>
  void write_blob(Iter iter, std::string prefix, dimz_t) const;
};

/// Alias for the simulator.
using mm_simulator_t = MultimaterialSimulator<>;

/// Alias for the simulator option manager.
using mm_sim_option_manager_t = typename mm_simulator_t::option_manager_t;

//==--- Implementation -----------------------------------------------------==//
//===== Public ----------------------------------------------------------=====//

template <typename... Ts>
void MultimaterialSimulator<Ts...>::simulate()
{
/*
  auto solver = solver_t{_data.input_iterator()};
  auto mat    = materials_t{};

  _params.print_static_summary();
  _data.initialize();

  auto cfl        = _params.cfl;
  auto wavespeeds = _data.wavespeed_iterator();
  auto timer      = util::default_timer_t();

  while (_params.continue_simulation())
  {
    //_params.cfl = _params.iters < 5 ? 0.18 : cfl;

    auto in  = _data.input_iterator();
    auto out = _data.output_iterator();

    // Set the wavespeed data based on the updated state data from the previous
    // iteration, and then update sim time delta based on max wavespeed:
    set_wavespeeds(in, wavespeeds, mat);
    _params.update_time_delta(max_element(_data.wavespeeds().begin(),      
                                          _data.wavespeeds().end()));
    //_params.print_current_status();

    solver.solve(in, out, mat, _params.dt_dh(), _setter);
    _params.update_simulation_info();
    _data.swap(out, in);
  }
  printf("Simulation time : %8lu ms\n", timer.elapsed_time());
  _params.print_final_summary();

  // Finalise the data, making sure it is all available on the host.
  _data.sync_device_to_host();
*/
}

// 


template <typename... Ts>
void MultimaterialSimulator<Ts...>::configure_cfl(double cfl)
{
  _params.cfl = cfl;
}



template <typename... Ts>
void MultimaterialSimulator<Ts...>::configure_resolution(double res)
{
  _params.domain.set_resolution(res);
}

template <typename... Ts>
void MultimaterialSimulator<Ts...>::configure_sim_time(double sim_time)
{
  _params.sim_time = sim_time;
}

template <typename... Ts>
void MultimaterialSimulator<Ts...>::configure_max_iterations(std::size_t iters)
{
  _params.max_iters = iters;
}

template <typename... Ts>
void MultimaterialSimulator<Ts...>::fill_data(fillinfo_container_t&& fillers)
{
/*
  using fill_state_t = typename traits_t::primitive_t;
  using index_t      = typename fill_state_t::index;
  auto& states       = _data.states();

  std::vector<int> indices = {};
  for (const auto& fillinfo : fillers) 
  {
    indices.emplace_back(index_t::from_name(fillinfo.data_name));
    if (indices.back() == -1)
    {
      throw std::runtime_error(
        std::string("State does not have a data element named: \n\t") +
        fillinfo.data_name
      );
    }
  }
*/
/*
  /// Go over each of the dimensions and fill the data:
  auto pos        = Array<float, 3>();
  auto dim_info   = dimension_info();
  auto fill_state = fill_state_t();
  auto material   = material_t();

  for (std::size_t i = 0; i < states.total_size(); ++i)
  {
    unrolled_for<dimensions>([&] (auto d)
    {
      constexpr auto dim = Dimension<d>{};
//      pos[d] = float(dim_info.flattened_index(i, dim)) / dim_info.size(dim);
      pos[d] = float(flattened_index(dim_info, i, dim)) / dim_info.size(dim);
    });

    // Invoke each of the fillers on the each state data property:
    std::size_t prop_index = 0;
    for (const auto& filler : fillers)
    {
      const auto value = filler.filler(pos);
      fill_state[prop_index++] = value;
    }
    states[i] = fill_state.conservative(material);
  }
*/
  // Make sure that the host and device data is
  // synced in the case that the data has GPU data.
  //_data.synchronize();
}

template <typename... Ts>
void MultimaterialSimulator<Ts...>::print_results() const
{
  std::ostream stream(nullptr);
  stream.rdbuf(std::cout.rdbuf());
  for_each(_mm_data, [&] (auto& mm_data)
  {
    stream_output_ascii(stream, mm_data);
  });
}

template <typename... Ts>
void MultimaterialSimulator<Ts...>::write_results(std::string prefix) const
{
  std::ofstream output_file;
  auto base_filename  = prefix + ".dat";
  auto material_index = int{1};
  for_each(_mm_data, [&] (auto& mm_data)
  {
    auto filename = std::string("material_") 
                  + std::to_string(material_index++)
                  + std::string("_")
                  + base_filename;
    output_file.open(filename, std::fstream::trunc);
    stream_output_ascii(output_file, mm_data);
    output_file.close(); 
  });
}

template <typename... Ts> template <typename Iter, typename Streams>
void MultimaterialSimulator<Ts...>::write_blob(Iter it, Streams& streams, dimx_t) const
{
/*
  for (const auto x : range(it.size(dim_x)))
  {
    unsigned i = 0;
    auto state = it.offset(x, dim_x)->primitive(material_t{});
    for (auto& stream : streams)
    {
      stream << state[i++] << " ";
    }
  }
  for (auto& stream : streams) stream << "\n";
*/
}

template <typename... Ts> template <typename Iter>
void MultimaterialSimulator<Ts...>::write_blob(Iter it, std::string prefix, dimx_t) const
{
/*
  using index_t = typename traits_t::primitive_t::index;
  std::vector<std::ofstream> streams;
  for (auto name : index_t::element_names())
  { 
    streams.emplace_back(prefix + "_" + name + "_" + std::to_string(_params.sim_time) + ".dat");
  }
  write_blob(it, streams, dim_x);
  for (auto& stream : streams) stream << "\n";
*/
}

template <typename... Ts> template <typename Iter>
void MultimaterialSimulator<Ts...>::write_blob(Iter it, std::string prefix, dimy_t) const
{
/*
  using index_t = typename traits_t::primitive_t::index;
  std::vector<std::ofstream> streams;
  for (auto name : index_t::element_names())
  { 
    streams.emplace_back(prefix + "_" + name + "_" + std::to_string(_params.sim_time) + ".dat");
  }
  for (const auto y : range(it.size(dim_y)))
  {
    write_blob(it.offset(y, dim_y), streams, dim_x);
  }
  for (auto& stream : streams) { stream.close(); }
*/
}

template <typename... Ts> template <typename Iter>
void MultimaterialSimulator<Ts...>::write_blob(Iter it, std::string prefix, dimz_t) const
{
/*
  for (const auto z : range(it.size(dim_z)))
  {
    auto new_prefix = prefix + "_" + std::to_string(z);
    write_blob(it.offset(z, dim_z), new_prefix, dim_y);
  }
*/
}

template <typename... Ts>
void MultimaterialSimulator<Ts...>::write_results_separate_raw(std::string prefix) const
{
/*
  write_blob(_data.states().multi_iterator(),
             prefix                   ,
             Dimension<dimensions-1>());
*/
}

//===== Private ---------------------------------------------------------=====//



template <typename... Ts> template <typename Stream, typename MatData>
void MultimaterialSimulator<Ts...>::stream_output_ascii(Stream&&       stream,
                                                        const MatData& data  ) const
{
  using index_t = typename traits_t::primitive_t::index;

  // Offset to convert dimensions {0, 1, 2} into char values {x, y, z}
  constexpr auto coord_char_offset = 120;
  const     auto comment           = std::string("# ");

  // Print the header:
  stream << comment << "t = " << std::to_string(_params.run_time) << "\n";
  int column = 1;
  unrolled_for<dimensions>([&] (auto dim)
  {
    constexpr auto coord_value = dim + coord_char_offset;
    stream << comment << "Column " << column++ << ": " 
           << char{coord_value}    << " coordinate\n";
  });
  for (const auto& element_name : index_t::element_names())
  {
    stream << comment << "Column " << column++ << ": " << element_name << "\n";
  }
  // Internal energy:
  stream << comment << "Column " << column << ": " << "internal energy (e)\n";

  auto append_element = [&stream] (auto element)
  {
    stream << std::setw(12)  << std::left           << std::fixed
           << std::showpoint << std::setprecision(8) << element << " ";
  };

  auto state_iterator = data.states().multi_iterator();
  auto eos            = data.material().eos();
  auto state          = state_iterator->primitive(eos);
  auto dim_info       = dimension_info();
  auto offsets        = std::array<std::size_t, dimensions>{0};
  do
  {
    unrolled_for<dimensions-1>([&] (auto i)
    {
      if (offsets[i] == dim_info.size(i))
      {
        offsets[i] = 0;
      }
    });

    for (const auto offset_x : range(dim_info.size(dim_x)))
    {
      state = state_iterator.offset(offset_x, dim_x)->primitive(eos);
      for (const auto dim : range(dimensions))
      {
        auto coord = (offsets[dim] + 0.5) * _params.domain.resolution();
        append_element(coord);
      }
      for (const auto& element : state)
      {
        append_element(element);
      }
      append_element(eos.eos(state));
      stream << "\n";
      offsets[dim_x]++;
    }
    unrolled_for<dimensions-1>([&] (auto i)
    {
      constexpr auto dim      = Dimension<i>{};
      constexpr auto next_dim = Dimension<i+1>{};
      if (offsets[dim] == dim_info.size(dim))
      {
        offsets[next_dim]++;
        state_iterator.shift(1, next_dim);
      }
    });
  } while (offsets[dimensions - 1] != dim_info.size(dimensions - 1));
}

}} // namespace fluid::sim

#endif // FLUIDITY_SIMULATOR_MULTIMATERIAL_SIMULATOR_HPP
