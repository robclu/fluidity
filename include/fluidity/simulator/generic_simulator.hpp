//==--- fluidity/simulator/generic_simulator.hpp ----------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  generic_simulator.hpp
/// \brief This file defines a generic simulator.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SIMULATOR_GENERIC_SIMULATOR_HPP
#define FLUIDITY_SIMULATOR_GENERIC_SIMULATOR_HPP

#include "parameters.hpp"
#include "simulation_data.hpp"
#include "simulation_traits.hpp"
#include "simulation_updater.hpp"
#include "simulator.hpp"
#include <fluidity/algorithm/fill.hpp>
#include <fluidity/algorithm/max_element.hpp>
#include <fluidity/dimension/dimension_info.hpp>
#include <fluidity/utility/type_traits.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <experimental/filesystem>

namespace fluid {
namespace sim   {

namespace fs = std::experimental::filesystem;

/// The GenericSimulator class implements the simulation interface.
/// \tparam Traits The traits which define the simulation paramters.
template <typename Traits>
class GenericSimulator final : public Simulator<Traits> {
 public:
  /// Defines the type of the traits class.
  using traits_t             = Traits;
  /// Defines the type of the base simulator class.
  using base_t               = Simulator<traits_t>;
  /// Defines the type of the filler container .
  using fillinfo_container_t = typename base_t::fillinfo_container_t;

 private:
  /// Defines the type of the state data to store, always conservative.
  using state_t     = typename traits_t::state_t;
  /// Defines the data type used in the state vector.
  using value_t     = typename state_t::value_t;
  /// Defines the type of the material for the simulation.
  using material_t  = typename traits_t::material_t;
  /// Defines execution policy for the simulator.
  using execution_t = typename traits_t::execution_t;
  /// Defines the type of the solver used to update the simulation.
  using solver_t    = typename traits_t::solver_t;
  /// Defines the type of the parameter container.
  using params_t    = Parameters<value_t>;
  /// Defines the type of the boundary setter.
  using setter_t    = solver::BoundarySetter;
  /// Defines the type of the data storage for the simulation.
  using storage_t   = SimulationData<traits_t, execution_t::device>;

  /// Defines a constexpr instance of the execution polcity.
  static constexpr auto execution_policy = execution_t{};

 public:
  /// Defines the number of spacial dimensions in the simulation.
  static constexpr auto dimensions = state_t::dimensions;

  /// Creates the simulator.
  GenericSimulator() : _params{dimensions} {}

  /// Cleans up all resources acquired by the simulator.
  ~GenericSimulator() {}

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
  base_t* configure_cfl(double cfl) override;

  /// Configures the simulator to set size and resolution of a dimension \p dim.
  /// \param[in] dim   The dimension to specify.
  /// \param[in] start The start value of the dimension.
  /// \param[in] end   The end value of the dimension.
  base_t* 
  configure_dimension(std::size_t dim, double start, double end) override;

  /// Configures the simulator to simulate for a maximum number of iterations.
  /// \param[in] iters  The maximum number of iterations to simulate for.
  base_t* configure_max_iterations(std::size_t iters) override;  

  /// Configures the simulator to use the \p resolution for the domain.
  /// \param[in] resolution The resolution to use for the domain.
  base_t* configure_resolution(double resolution) override;

  /// Configures the simulator to simulate until a certain simulation time.
  /// \param[in] sim_time The time to run the simulation until.
  base_t* configure_sim_time(double sim_time) override;

  /// Prints the results of the simulation to the standard output stream.
  void print_results() const override;

  /// Writes the results of the simulation to the \p path using the \p prefix
  /// appended to the property which is output. If \p path = "", then the
  /// current working directory is used as the path.
  /// \param[in] file_path   The path (inluding the file prefix) to write to.
  void write_results(std::string file_path) const override;

 private:
  /// Defines a constexpr instance of a tag which is std::true_type of the batch
  /// size must be fetched for 1 spacial dimension.
  static constexpr auto batch_size_tag = 
    std::integral_constant<bool, traits_t::spacial_dims == 1>{};

  storage_t _data;    //!< Data for the simulation.
  params_t  _params;  //!< The parameters for the simulation.
  setter_t  _setter;  //!< The boundary setter.

  /// Returns the dimension information for the simulator.
  auto dimension_info() const;

  /// Implementation of the outputting functionality. The \p stream parameter
  /// is used to determine if the output is written to a file or if it is
  /// sent to the standard output stream.
  /// 
  /// This implementation will print the data in columns, with a column for the
  /// position of the cell in each dimension, and then columns for each element
  /// in the state vector.
  /// 
  /// \param[in] stream   The stream to output the results to.
  /// \tparam    Stream   The type of the output stream.
  template <typename Stream>
  void stream_output_ascii(Stream&& stream) const;
};

//==--- Implementation -----------------------------------------------------==//
//===== Public ----------------------------------------------------------=====//

template <typename Traits>
void GenericSimulator<Traits>::simulate()
{
  using namespace std::chrono;

  // Variables for debug info:
  auto start = high_resolution_clock::now();
  auto end   = high_resolution_clock::now();

  // Variables for simulation specification:
  auto threads = get_thread_sizes(_data.input_iterator());
  auto blocks  = get_block_sizes(_data.input_iterator(), threads);
  auto solver  = solver_t{};
  auto mat     = material_t{};

  _params.print_static_summary();

  // Initialize the data:
  _data.initialize();

  auto cfl = _params.cfl;
  while (_params.continue_simulation())
  {
    _params.cfl = _params.iters < 5 ? 0.18 : cfl;

    auto input_it     = _data.input_iterator();
    auto output_it    = _data.output_iterator();
    auto wavespeed_it = _data.wavespeed_iterator();

    _params.print_current_status();

    // Set the wavespeed data based on the updated state data from the previous
    // iteration, and then update sim time delta based on max wavespeed:
    set_wavespeeds(input_it, wavespeed_it, mat);
    _params.update_time_delta(
      max_element(_data.wavespeeds().begin(),_data.wavespeeds().end()));

    update(input_it       ,
           output_it      ,
           solver         ,
           mat            ,
           _params.dt_dh(),
           threads        ,
           blocks         ,
           _setter        );
    _params.update_simulation_info();
    _data.swap_states();

    // Debugging ...
    //_data.finalise_states();
    // If debugging, set option to print based on iterations check:
    //std::string filename = "Debug_" + std::to_string(_params.iters);
    //this->write_results(filename);
  }

  end = high_resolution_clock::now();
  printf("Simulation time : %8lu ms\n", 
    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

  // Finalise the data, making sure it is all available on the host.
  _data.finalise();
}

template <typename Traits>
Simulator<Traits>*
GenericSimulator<Traits>::configure_cfl(double cfl)
{
  _params.cfl = cfl;
  return this;
}

template <typename Traits>
Simulator<Traits>*
GenericSimulator<Traits>::configure_dimension(std::size_t dim  ,
                                              double      start,
                                              double      end  )
{
  _params.domain.set_dimension(dim, start, end);
  _data.resize_dim(dim, _params.domain.elements(dim));
  return this;
}

template <typename Traits>
Simulator<Traits>* GenericSimulator<Traits>::configure_resolution(double res)
{
  _params.domain.set_resolution(res);
  return this;
}

template <typename Traits>
Simulator<Traits>* GenericSimulator<Traits>::configure_sim_time(double sim_time)
{
  _params.sim_time = sim_time;
  return this;
}

template <typename Traits>
Simulator<Traits>*
GenericSimulator<Traits>::configure_max_iterations(std::size_t iters)
{
  _params.max_iters = iters;
  return this;
}

template <typename Traits>
void GenericSimulator<Traits>::fill_data(fillinfo_container_t&& fillers)
{
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
      pos[d] = float(dim_info.flattened_index(i, dim)) / dim_info.size(dim);
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

  // Make sure that the host and device data is
  // synced in the case that the data has GPU data.
  _data.synchronize();
}

template <typename Traits>
void GenericSimulator<Traits>::print_results() const
{
  std::ostream stream(nullptr);
  stream.rdbuf(std::cout.rdbuf());
  stream_output_ascii(stream);
}

template <typename Traits>
void GenericSimulator<Traits>::write_results(std::string prefix) const
{
  std::ofstream output_file;
  auto filename = prefix + ".dat";
  output_file.open(filename, std::fstream::trunc);
  stream_output_ascii(output_file);
  output_file.close(); 
}

//===== Private ---------------------------------------------------------=====//

template <typename Traits>
auto GenericSimulator<Traits>::dimension_info() const
{
  auto dim_info = DimInfo<dimensions>();
  unrolled_for<dimensions>([&] (auto i)
  {
    dim_info[i] = _data.states().size(i);
  });
  return dim_info;
}

template <typename Traits> template <typename Stream>
void GenericSimulator<Traits>::stream_output_ascii(Stream&& stream) const
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
  stream << comment << "Column " << column << " : " << "internal energy (e)\n";

  auto append_element = [&stream] (auto element)
  {
    stream << std::setw(12)  << std::left           << std::fixed
           << std::showpoint << std::setprecision(8) << element << " ";
  };

  auto state_iterator = _data.states().multi_iterator();
  auto material       = material_t();
  auto state          = state_iterator->primitive(material);
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
      state = state_iterator.offset(offset_x, dim_x)->primitive(material);
      for (const auto dim : range(dimensions))
      {
        auto coord = (offsets[dim] + 0.5) * _params.domain.resolution();
        append_element(coord);
      }
      for (const auto& element : state)
      {
        append_element(element);
      }
      append_element(material.eos(state));
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

#endif // FLUIDITY_SIMULATOR_GENERIC_SIMULATOR_HPP