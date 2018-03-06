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

#include "simulation_traits.hpp"
#include "simulator.hpp"
#include <fluidity/container/host_tensor.hpp>
#include <fluidity/dimension/dimension_info.hpp>
#include <iostream>

namespace fluid {
namespace sim   {

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
  /// Defines the type of the state data to store.
  using state_t   = typename traits_t::state_t;
  /// Defines the type of the container used to store the state state.
  using storage_t = HostTensor<state_t, state_t::dimensions>;

 public:
  /// Defines the number of spacial dimensions in the simulation.
  static constexpr auto dimensions = state_t::dimensions;

  /// Creates the simulator.
  GenericSimulator() {};

  /// Cleans up all resources acquired by the simulator.
  ~GenericSimulator() {}

  /// Runs the simulation until completion.
  void simulate() override;

  /// Fills the simulator with simulation data for a simulator, where each of
  /// the fillers store the name of the property which they are filling, and 
  /// a function object which can fill values based on their position in the
  /// space.
  /// \param[in] fillers A container of fillers for filling the data.
  virtual void fill_data(fillinfo_container_t&& fillers) override;

  /// Prints the results of the simulation to the standard output stream.
  void print_results() const override;

  /// Writes the results of the simulation to the \p path using the \p prefix
  /// appended to the property which is output. If \p path = "", then the
  /// current working directory is used as the path.
  /// \param[in] file_path   The path (inluding the file prefix) to write to.
  void write_results(fs::path file_path) const override;

 private:
  storage_t _initial_states;  //!< States at the start of an iteration.
  storage_t _updated_states;  //!< States at the end of an iteration.

  /// Returns the dimension information for the simulator.
  DimInfo dimension_info() const;

  /// Implementation of the outputting functionality. The \p stream parameter
  /// is used to determine if the output is written to a file or if it is
  /// sent to the standard output stream.
  /// \param[in] stream   The stream to output the results to.
  /// \tparam    Stream   The type of the output stream.
  template <typename Stream>
  void stream_output(Stream&& stream) const ;
};

//==--- Implementation -----------------------------------------------------==//
//===== Public ----------------------------------------------------------=====//

template <typename Traits>
void GenericSimulator<Traits>::simulate()
{

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
  for (int i = 0; i < _initial_states.size(); ++i)
  {
    unrolled_for<dimensions>([&] (auto d)
    {
      constexpr auto dim = Dimension<d>{};
      pos[d] = float(dim_info.flattened_index(i, dim)) / dim_info.size(dim);
    });

    // Invoke each of the fillers on the each state data property:
    for (const auto& filler : fillers)
    {
      for (auto prop_index : indices)
      {
        _initial_states[i][prop_index] = filler.filler(pos);
      }
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
DimInfo GenericSimulator<Traits>::dimension_info() const
{
  auto dim_info = DimInfo();
  unrolled_for<dimensions>([&] (auto i)
  {
    dim_info.push_back(_initial_states.size(i));
  });
  return dim_info;
}

template <typename Traits> template <typename Stream>
void GenericSimulator<Traits>::stream_output(Stream&& stream) const
{
  using index_t = typename state_t::index;

  // If the data is 1 or 2 dimensional, then there is no offset, otherwise
  // an offset is created for the page of 2D data which is being output.
  constexpr auto page_number = dimensions <= 2 ? 0 : 3;

  // We iterate over all dimensions past the first 2, and then output 2D pages
  // for each of that data.
  constexpr auto iterations = dimensions <= 2 ? 1 : dimensions - 2;

  auto dim_info = dimension_info();
  unrolled_for<iterations>([&, this] (auto dim)
  {
    const auto batch_size = dim_info.size(dim_x)
                          * (dimensions == 1 ? 1 : dim_info.size(dim_y));

    // For each of the state elements:
    const auto elements = index_t::element_names();
    for (const auto element_idx : range(elements.size()))
    {
      for (const auto outer_idx : range(dim_info.size(dim)))
      {
        // Create the filename / header, which has the form:
        // <element_name>_<page_number+dim>_<index in dimension>;
        std::string output = elements[element_idx]             + "_"
                           + std::to_string(page_number + dim) + "_"
                           + std::to_string(outer_idx);

        if constexpr (std::is_same_v<fs::path, std::decay_t<Stream>>)
        {
          stream /= output + ".txt";
        }
        else
        {
          stream << output << ":\n\n";
        }

        const auto offset = outer_idx * dim_info.offset(Dimension<dim>{});
        for (const auto inner_idx : range(batch_size))
        {
          stream << _updated_states[offset + inner_idx]
                 << inner_idx % dim_info.size(dim_x) == 0 ? "\n" : " ";
        }
      }
    }
  }); 
}

}} // namespace fluid::sim

#endif // FLUIDITY_SIMULATOR_GENERIC_SIMULATOR_HPP