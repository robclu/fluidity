//==--- fluidity/simulator/simulator_impl.hpp -------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  simulator_impl.hpp
/// \brief This file provides an implementation of the simulator interface.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SIMULATOR_SIMULATOR_IMPL_HPP
#define FLUIDITY_SIMULATOR_SIMULATOR_IMPL_HPP

#include "parameters.hpp"
#include "simulator.hpp"
#include "simulation_data.hpp"
#include "simulator_traits.hpp"
#include "wavespeed_initialization.hpp"
#include <fluidity/algorithm/fill.hpp>
#include <fluidity/algorithm/max_element.hpp>
#include <fluidity/dimension/dimension_info.hpp>
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

/// The SimulatorImpl class implements the simulation interface, as well as the
/// configurable interface to allow the configuration of the simulation.
/// \tparam Traits The traits which define the simulation parameters.
template <typename... Traits>
class SimulatorImpl : 
  public Simulator  ,
  public setting::Configurable<SimulatorImpl<Traits...>> {
 public:
  /// Allows the OptionManager class to build the simulator implementation.
  /// \tparam Ts The traits to make the simulator implementation with.
  template <typename... Ts>
  using make_type_t = SimulatorImpl<Ts...>;

  /// Defines the type of the traits class.
  //using traits_t             = Traits;
  /// Defines the type of the base simulator class.
  using base_t               = Simulator;
  /// Defines the type of this class.
  using this_t               = SimulatorImpl;
  /// Defines the type of the filler container .
  using fillinfo_container_t = typename base_t::fillinfo_container_t;

 private:
  /// Defines the type of the traits for the simulation implementation.
  using traits_t = SimTraits<base_t, this_t, Traits...>;

  /// Defines the type of the state for the simulation.
  using state_t      = typename traits_t::state_t;
  /// Defines the data type used in the state vector.
  using value_t      = typename traits_t::data_t;
  /// Defines the type of the material for the simulation.
  using material_t   = typename traits_t::material_t;
  /// Defines the type of the reconstruction method to use.
  using recon_t      = typename traits_t::recon_t;
  /// Defines execution policy for the simulator.
  using exec_t       = typename traits_t::exec_t;
  /// Defines the type of the solver.
  using solver_t     = typename traits_t::solver_t;

  /// Defines the type of the parameter container.
  using params_t    = Parameters<value_t>;
  /// Defines the type of the boundary setter.
  using setter_t    = solver::BoundarySetter;
  /// Defines the type of the data storage for the simulation.
  using storage_t   = SimulationData<traits_t, exec_t::device>;

  /// Defines a constexpr instance of the execution polcity.
  static constexpr auto execution_policy = exec_t{};

 public:
  /// Defines the number of spacial dimensions in the simulation.
  static constexpr auto dimensions = traits_t::dimensions;

  /// Defines the type of the option manager which is used to configure the
  /// types to use for the simulation.
  using option_manager_t = typename traits_t::option_manager_t;

  /// Creates the simulator.
  SimulatorImpl() : _params{dimensions} {}

  /// Cleans up all resources acquired by the simulator.
  ~SimulatorImpl() {}

  /// Runs the simulation until completion.
  void simulate() override;

/*
  void configure(const setting::Parameter* param) override
  {
    std::cout << "Don't know how to set this parameter\n";
  }
*/
/*
  /// Configures the simulator from using the paramter manager.
  void configure(const setting::CflParameter* cfl_param) override
  {
    _params.cfl = cfl_param->cfl();
    std::cout << "Set sim cfl param : " << _params.cfl << "\n";
  }
*/
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
  void configure_dimension(std::size_t dim, double start, double end) override;

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

 private:
  /// Defines a constexpr instance of a tag which is std::true_type of the batch
  /// size must be fetched for 1 spacial dimension.
  static constexpr auto batch_size_tag = 
    std::integral_constant<bool, traits_t::dimensions == 1>{};

  storage_t _data;    //!< Data for the simulation.
  params_t  _params;  //!< The parameters for the simulation.
  setter_t  _setter;  //!< The boundary setter.



/*
  void configure(const setting::Parameter* param)
  {
    std::cout << "Don't know how to set this param : " << param->type() << "\n";
  }
*/

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
using simulator_t = SimulatorImpl<>;

/// Alias for the simulator option manager.
using sim_option_manager_t = typename simulator_t::option_manager_t;

//==--- Implementation -----------------------------------------------------==//
//===== Public ----------------------------------------------------------=====//

template <typename... Ts>
void SimulatorImpl<Ts...>::simulate()
{
  auto solver = solver_t{_data.input_iterator()};
  auto mat    = material_t{};

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
}

template <typename... Ts>
void SimulatorImpl<Ts...>::configure_cfl(double cfl)
{
  _params.cfl = cfl;
}

template <typename... Ts>
void SimulatorImpl<Ts...>::configure_dimension(std::size_t dim  ,
                                                     double      start,
                                                     double      end  )
{
  _params.domain.set_dimension(dim, start, end);
  _data.resize_dim(dim, _params.domain.elements(dim));
}

template <typename... Ts>
void SimulatorImpl<Ts...>::configure_resolution(double res)
{
  _params.domain.set_resolution(res);
}

template <typename... Ts>
void SimulatorImpl<Ts...>::configure_sim_time(double sim_time)
{
  _params.sim_time = sim_time;
}

template <typename... Ts>
void SimulatorImpl<Ts...>::configure_max_iterations(std::size_t iters)
{
  _params.max_iters = iters;
}

template <typename... Ts>
void SimulatorImpl<Ts...>::fill_data(fillinfo_container_t&& fillers)
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

  // Make sure that the host and device data is
  // synced in the case that the data has GPU data.
  _data.synchronize();
}

template <typename... Ts>
void SimulatorImpl<Ts...>::print_results() const
{
  std::ostream stream(nullptr);
  stream.rdbuf(std::cout.rdbuf());
  stream_output_ascii(stream);
}

template <typename... Ts>
void SimulatorImpl<Ts...>::write_results(std::string prefix) const
{
  std::ofstream output_file;
  auto filename = prefix + ".dat";
  output_file.open(filename, std::fstream::trunc);
  stream_output_ascii(output_file);
  output_file.close(); 
}

template <typename... Ts> template <typename Iter, typename Streams>
void SimulatorImpl<Ts...>::write_blob(Iter it, Streams& streams, dimx_t) const
{
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
}

template <typename... Ts> template <typename Iter>
void SimulatorImpl<Ts...>::write_blob(Iter it, std::string prefix, dimx_t) const
{
  using index_t = typename traits_t::primitive_t::index;
  std::vector<std::ofstream> streams;
  for (auto name : index_t::element_names())
  { 
    streams.emplace_back(prefix + "_" + name + "_" + std::to_string(_params.sim_time) + ".dat");
  }
  write_blob(it, streams, dim_x);
  for (auto& stream : streams) stream << "\n";
}

template <typename... Ts> template <typename Iter>
void SimulatorImpl<Ts...>::write_blob(Iter it, std::string prefix, dimy_t) const
{
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
}

template <typename... Ts> template <typename Iter>
void SimulatorImpl<Ts...>::write_blob(Iter it, std::string prefix, dimz_t) const
{
  for (const auto z : range(it.size(dim_z)))
  {
    auto new_prefix = prefix + "_" + std::to_string(z);
    write_blob(it.offset(z, dim_z), new_prefix, dim_y);
  }
}

template <typename... Ts>
void SimulatorImpl<Ts...>::write_results_separate_raw(std::string prefix) const
{
  write_blob(_data.states().multi_iterator(),
             prefix                   ,
             Dimension<dimensions-1>());
}

//===== Private ---------------------------------------------------------=====//

template <typename... Ts>
auto SimulatorImpl<Ts...>::dimension_info() const
{
  auto dim_info = DimInfo<dimensions>();
  unrolled_for<dimensions>([&] (auto i)
  {
    dim_info[i] = _data.states().size(i);
  });
  return dim_info;
}

template <typename... Ts> template <typename Stream>
void SimulatorImpl<Ts...>::stream_output_ascii(Stream&& stream) const
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

#endif // FLUIDITY_SIMULATOR_SIMULATOR_IMPL_HPP
