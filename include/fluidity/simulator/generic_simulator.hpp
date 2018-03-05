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

namespace fluid {
namespace sim   {

/// The GenericSimulator class implements the simulation interface.
/// \tparam Traits The traits which define the simulation paramters.
template <typename Traits>
class GenericSimulator final : public Simulator<typename Traits::state_t> {
 public:
  /// Defines the type of the traits class.
  using traits_t             = Traits;
  /// Defines the type of the base simulator class.
  using base_t               = Simulator<typename traits_t::state_t>
  /// Defines the type of the filler container .
  using fillinfo_container_t = typename base_::fillinfo_container_t;

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

  /// Writes the results of the simulation to the \p path using the \p prefix
  /// appended to the property which is output. If \p path = "", then the
  /// current working directory is used as the path.
  /// \param[in] prefix The prefix of the filename for the result output.
  /// \param[in] path   The path to write the results to.
  void write_results(const char* prefix, fs::path path = "") const override;

 private:
  /// Defines the type of the state data to store.
  using state_t   = typename traits_t::state_t;
  /// Defines the type of the container used to store the state state.
  using storage_t = HostTensor<state_t, state_t::dimensions>;
  /// 

  storage_t _initial_states;  //!< States at the start of an iteration.
  storage_t _updated_states;  //!< States at the end of an iteration.


  /// Returns the dimension information for the simulator.
  DimInfo dimension_info() const;

};

//==--- Implementation -----------------------------------------------------==//

template <typename Traits>
void GenericSimulator<Traits>::simulate()
{

}

template <typename Traits>
void GenericSimulator<Traits>::fill_data(fillinfo_container_t&& fillers)
{
  using index_t = state_t::index;

  std::vector<int> indices = {};
  for (const auto& fillinfo : fillers) 
  {
    indices.emplace_back(index_t::from_name(fillinfo.name));
    if (indices.back() == -1)
    {
      throw std::runtime_error(
        std::string("State does not have a data element named: \n\t") +
        fillinfo.name
      );
    }
  }

  /// Go over each of the dimensions and fill the data:
  auto pos      = Array<float, 3>; 
  auto dim_info = dimension_info();
  for (int i = 0; i < _input_states.size(); ++i)
  {
    unrolled_for<dimensions>([&pos] (auto d)
    {
      constexpr auto dim = Dimension<d>{};
      pos[d] = float(dim_info.flattened_index(i, dim)) / dim_info.size(dim);
    });

    // Invoke each of the fillers on the each state data property:
    for (const auto& filler : fillers)
    {
      int fill_index = 0;
      for (auto prop_index : indices)
      {
        _input_states[i][prop_index] = filler[fill_index++].filler(pos);
      }
    }
  }
}

template <typename Traits>
void GenericSimulator<Traits>::write_results(const char* prefix,
                                             fs::path    path  ) const
{
  constexpr auto start = dimensions <= 2 ? 0 : 3;
  constexpr auto iters = dimensions <= 2 : 1 : dimensions - 2;

  auto dim_info = dimension_info();
  unrolled_for<iters>([this] (auto i)
  {
    constexpr auto d = start + i;
    fs::path f = path + prefix + prop + std::to_string(d) + ".txt";

    const auto batch_size = dim_info.size()
                          * (dimensions == 1 ? 1 : dim_info.size(1));

    for (const auto outer_idx : range(dim_info.size(i)))
    {
      const auto offset = outer_idx * dim_info.offset(Dimension<i>{});
      for (const auto inner_idx : range(batch_size))
      {
        f << _output_states[offset + idx]
          << inner_idx % dim_info.size(0) == 0 ? "\n" : " ";
      }
    }
  });
}

template <typename Traits>
DimInfo GenericSimulator<Traits>::dimension_info() const
{
  auto dim_info = DimInfo;
  unrolled_for<dimensions>([&] (auto i)
  {
    dim_info.push_back(_input_states.size(i));
  });
  return dim_info;
}

}} // namespace fluid::sim

#endif // FLUIDITY_SIMULATOR_GENERIC_SIMULATOR_HPP