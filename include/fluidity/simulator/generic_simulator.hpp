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
#include "simulation_traits.hpp"
#include "simulation_updater.hpp"
#include "simulator.hpp"
#include <fluidity/algorithm/fill.hpp>
#include <fluidity/algorithm/max_element.hpp>
#include <fluidity/container/device_tensor.hpp>
#include <fluidity/container/host_tensor.hpp>
#include <fluidity/dimension/dimension_info.hpp>
#include <fluidity/execution/execution_policy.hpp>
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
  /// Defines the type used for dimension information specification.
  using dim_spec_t           = typename base_t::DimSpec;

 private:
  /// Defines the type of the state data to store.
  using state_t     = typename traits_t::state_t;
  /// Defines the data type used in the state vector.
  using value_t     = typename state_t::value_t;
  /// Defines the type of the material for the simulation.
  using material_t  = typename traits_t::material_t;
  /// Defines execution policy for the simulator.
  using execution_t = typename traits_t::execution_t;
  /// Defines the type of the container used to store the state state.
  using storage_t   = HostTensor<state_t, state_t::dimensions>;
  /// Defines the type of the solver used to update the simulation.
  using solver_t    = typename traits_t::solver_t;
  /// Defines the type of the container used for storing wavespeed data.
  using wavespeed_t = HostTensor<value_t, 1>;
  /// Defines the type of the parameter container.
  using params_t    = Parameters<value_t>;

  /// Defines a constexpr instance of the execution polcity.
  static constexpr auto execution_policy = execution_t{};

 public:
  /// Defines the number of spacial dimensions in the simulation.
  static constexpr auto dimensions = state_t::dimensions;

  /// Creates the simulator.
  GenericSimulator() {}

  /// Cleans up all resources acquired by the simulator.
  ~GenericSimulator() 
  {
    printf("Destroying simulator!\n");
  }

  /// Runs the simulation until completion.
  void simulate() override;

  /// Fills the simulator with simulation data for a simulator, where each of
  /// the fillers store the name of the property which they are filling, and 
  /// a function object which can fill values based on their position in the
  /// space.
  /// \param[in] fillers A container of fillers for filling the data.
  void fill_data(fillinfo_container_t&& fillers) override;

  /// Configures the simulator to set size and resolution of a dimension \p dim.
  /// \param[in] dim  The dimension to specify.
  /// \param[in] spec The specification of the dimension.
  base_t* configure_dimension(std::size_t dim, dim_spec_t spec) override; 

  /// Configures the simulator to simulate until a certain simulation time.
  /// \param[in] sim_time The time to run the simulation until.
  base_t* configure_sim_time(double sim_time) override;

  /// Prints the results of the simulation to the standard output stream.
  void print_results() const override;

  /// Writes the results of the simulation to the \p path using the \p prefix
  /// appended to the property which is output. If \p path = "", then the
  /// current working directory is used as the path.
  /// \param[in] file_path   The path (inluding the file prefix) to write to.
  void write_results(fs::path file_path) const override;

 private:
  /// Defines a constexpr instance of a tag which is std::true_type of the batch
  /// size must be fetched for 1 spacial dimension.
  static constexpr auto batch_size_tag = 
    std::integral_constant<bool, traits_t::spacial_dims == 1>{};

  storage_t   _initial_states;  //!< States at the start of an iteration.
  storage_t   _updated_states;  //!< States at the end of an iteration.
  wavespeed_t _wavespeeds;      //!< Wavespeeds for the simulation.
  params_t    _params;          //!< The parameters for the simulation.

  /// Returns the dimension information for the simulator.
  decltype(auto) dimension_info() const;

  /// Outputs a batch of 1D or 2D data to the \p stream where the batch contains
  /// \p batch_size number of total elements. The \p offset defines the offset
  /// into the state container in 1D (flattened) form, and the element which is
  /// output is the element at position \p element_index in the state vector.
  /// \param[in] stream        The stream to output the element to.
  /// \param[in] offset        The offset of the first element in the batch into
  ///            the state container.
  /// \param[in] batch_size    The size of the batch to output.
  /// \param[in] element_index The index of the element in the state container.
  template <typename Stream>
  void output_batch(Stream&&    stream       ,
                    std::size_t offset       ,
                    std::size_t batch_size   ,
                    std::size_t element_index) const;

  /// Implementation of the outputting functionality. The \p stream parameter
  /// is used to determine if the output is written to a file or if it is
  /// sent to the standard output stream.
  /// \param[in] stream   The stream to output the results to.
  /// \tparam    Stream   The type of the output stream.
  template <typename Stream>
  void stream_output(Stream&& stream) const;

  void output_data(std::ostream& output_stream,
                        std::string   output       ,
                        std::size_t   offset       ,
                        std::size_t   batch_size   ,
                        std::size_t   element_idx  ) const;

  void output_data(fs::path                 ,
                      std::string output       ,
                      std::size_t offset       ,
                      std::size_t batch_size   ,
                      std::size_t element_idx  ) const;


  auto& get_sim_input_states(exec::cpu_type) const
  {
    return _initial_states;
  }

  auto get_sim_input_states(exec::gpu_type) const
  {
    return std::move(_initial_states.as_device());
  } 

  auto& get_sim_output_states(exec::cpu_type) const
  {
    return _updated_states;
  }

  auto get_sim_output_states(exec::gpu_type) const
  {
    return std::move(_updated_states.as_device());
  }

  auto& get_sim_wavespeeds(exec::cpu_type) const
  {
    return _wavespeeds;
  }

  auto get_sim_wavespeeds(exec::gpu_type) const
  {
    return std::move(_wavespeeds.as_device());
  }

  template <typename States>
  void finalise_output_states(const States& states, exec::cpu_type) {}

  template <typename States>
  void finalise_output_states(const States& states, exec::gpu_type)
  {
    _updated_states = states.as_host();
  }

  constexpr std::size_t get_batch_size(std::true_type) const
  {
    return _updated_states.size(dim_x);
  }

  constexpr std::size_t get_batch_size(std::false_type) const
  {
    return dimension_info().size(dim_y);
  }
};

//==--- Implementation -----------------------------------------------------==//
//===== Public ----------------------------------------------------------=====//

template <typename Traits>
void GenericSimulator<Traits>::simulate()
{
  using namespace std::chrono;
  auto time  = double{0.0};
  auto iters = std::size_t{0};
  auto start = high_resolution_clock::now();
  auto end   = high_resolution_clock::now();

  auto input_states  = get_sim_input_states(execution_policy);
  auto output_states = get_sim_output_states(execution_policy);
  auto wavespeeds    = get_sim_wavespeeds(execution_policy);

  auto input_it  = input_states.multi_iterator();
  auto output_it = output_states.multi_iterator();
  auto wavespeed_it = wavespeeds.multi_iterator();

  auto threads   = get_thread_sizes(input_it);
  auto blocks    = get_block_sizes(input_it, threads);
  auto solver    = solver_t{};
  auto mat       = material_t{};

#if !defined(NDEBUG)
  printf("PARAMETERS      \n----------------\n");
  printf("RUN TIME  : %5.5f\n", _params.run_time);
  printf("MAX ITERS : %5lu\n", _params.max_iters);
  printf("----------------\n\n");

  printf("ITERATION | TIME\n----------------\n");
#endif

  while (time < _params.run_time && iters < _params.max_iters)
  {
#if !defined(NDEBUG)
    printf("%10lu|%5.5f\n", iters, time);
#endif // NDEBUG

    input_it     = input_states.multi_iterator();
    output_it    = output_states.multi_iterator();
    wavespeed_it = wavespeeds.multi_iterator();

    set_wavespeeds(input_it, wavespeed_it, mat);

    printf("A : %4.4f\n", max_element(wavespeeds.begin(), wavespeeds.end()));
    _params.update(max_element(wavespeeds.begin(), wavespeeds.end()));

    // Set boundary ghost cells ...
    // Set patch ghost cells ...
   
    printf("B\n");
    // Update the simulation ...
    update(input_it, output_it, solver, mat, _params.dt_dh(), threads, blocks);

    time  += _params.dt();
    iters += 1;
    printf("C\n");

    std::swap(input_states, output_states);
    printf("D\n");
  }

  finalise_output_states(output_states, execution_policy);
}

template <typename Traits>
Simulator<Traits>*
GenericSimulator<Traits>::configure_dimension(std::size_t /*dim*/, 
                                              dim_spec_t  spec   )
{
  _initial_states.resize(spec.elements());
  _updated_states.resize(spec.elements());
  _wavespeeds.resize(spec.elements());
  return this;
}

template <typename Traits>
Simulator<Traits>* GenericSimulator<Traits>::configure_sim_time(double sim_time)
{
  _params.run_time = sim_time;
  return this;
}

template <typename Traits>
void GenericSimulator<Traits>::fill_data(fillinfo_container_t&& fillers)
{
  using index_t = typename state_t::index;

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
  auto pos      = Array<float, 3>();
  auto dim_info = dimension_info();
  for (std::size_t i = 0; i < _initial_states.total_size(); ++i)
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
      _initial_states[i][prop_index]   = value;
      _updated_states[i][prop_index++] = value;
    }
  }
}

template <typename Traits>
void GenericSimulator<Traits>::print_results() const
{
  std::ostream stream(nullptr);
  stream.rdbuf(std::cout.rdbuf());
  stream_output(stream);
}


template <typename Traits>
void GenericSimulator<Traits>::write_results(fs::path file_path) const
{
  stream_output(file_path);
}

//===== Private ---------------------------------------------------------=====//

template <typename Traits>
decltype(auto) GenericSimulator<Traits>::dimension_info() const
{
  auto dim_info = DimInfo<dimensions>();
  unrolled_for<dimensions>([&] (auto i)
  {
    dim_info[i] = _initial_states.size(i);
  });
  return dim_info;
}

template <typename Traits> template <typename Stream>
void GenericSimulator<Traits>::output_batch(Stream&&    stream       ,
                                            std::size_t offset       ,
                                            std::size_t batch_size   ,
                                            std::size_t element_index) const
{
  auto dim_info = dimension_info();
  for (const auto index : range(batch_size))
  {
    stream << std::setw(8) << std::right
           << std::fixed   << std::showpoint << std::setprecision(4)
           << _updated_states[offset + index][element_index]
           << ((index % dim_info.size(dim_x) == 0 && index != 0) ? "\n" : " ");
  }
}

template <typename Traits> template <typename Stream>
void GenericSimulator<Traits>::stream_output(Stream&& stream) const
{
  using index_t = typename state_t::index;

  // We iterate over all dimensions past the first 2, and then output 2D pages
  // for each of that data.
  constexpr auto iterations = dimensions <= 2 ? 1 : dimensions - 2;

  auto dim_info = dimension_info();
  unrolled_for<iterations>([&, this] (auto dim)
  {
    const auto element_names = index_t::element_names();
    const auto batch_size    = get_batch_size(batch_size_tag);

    for (const auto element_idx : range(element_names.size()))
    {
      constexpr auto dim_iterations = dimensions <= 2 ? 1 : dim_info.size(dim);

      // Iterate over the elements in the dimension, and output a page (2D)
      // matrix of data for that portion of data ...
      for (const auto dim_idx : range(dim_iterations))
      {
        // If the data is 1 or 2 dimensional, then there is no offset, otherwise
        // an offset is created for the page of 2D data which is being output.
        constexpr auto page_number = dimensions <= 2 ? 0 : 3;

        // Create the filename / header, which has the form:
        // <element_name>_<page_number+dim>_<index in dimension>;
        std::string left   = "_<", comma = ",", right = ">";
        std::string output = element_names[element_idx]        + left
                           + "p-" + std::to_string(page_number + dim) + comma
                           + "d-" + std::to_string(dim_idx)           + right;

        const auto offset = dim_idx * dim_info.offset(Dimension<dim>{});
        output_data(stream, output, offset, batch_size, element_idx);
      }
    }
  }); 
}

template <typename Traits>
void GenericSimulator<Traits>::output_data(std::ostream& output_stream,
                                                std::string   output       ,
                                                std::size_t   offset       ,
                                                std::size_t   batch_size   ,
                                                std::size_t   element_idx  ) const
{
  output_stream << output << "\n";
  output_batch(output_stream, offset, batch_size, element_idx);
  output_stream << "\n";  
}

template <typename Traits>
void GenericSimulator<Traits>::output_data(fs::path                 ,
                                              std::string output       ,
                                              std::size_t offset       ,
                                              std::size_t batch_size   ,
                                              std::size_t element_idx  ) const
{
  std::ofstream output_file;
  output_file.open(output += ".txt", std::fstream::app);
  output_batch(output_file, offset, batch_size, element_idx);
  output_file.close();
}

}} // namespace fluid::sim

#endif // FLUIDITY_SIMULATOR_GENERIC_SIMULATOR_HPP