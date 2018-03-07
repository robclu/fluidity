//==--- fluidity/simulator/simulator.hpp ------------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  simulator.hpp
/// \brief This file defines the interface for simulation objects.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SIMULATOR_SIMULATOR_HPP
#define FLUIDITY_SIMULATOR_SIMULATOR_HPP

#include <functional>
#include <vector>
#include <experimental/filesystem>

namespace fluid {
namespace sim   {

// Alias for the filesystem.
namespace fs = std::experimental::filesystem;

/// The Simulator class defines the interface for simulation objects.
/// \tparam Traits The traits of the simulator.
template <typename Traits>
class Simulator {
 public:
  /// Defines the type of this simulator.
  using self_t   = Simulator<Traits>;
  /// Defines the traits of the simulation.
  using traits_t = std::decay_t<Traits>;
  /// Defines the type of the state which is used for the simulation.
  using state_t  = typename traits_t::state_t;
  /// Defines the type of the data used by the state.
  using value_t  = typename state_t::value_t;
  /// Defines the type of the functor which can be used to fill elements.
  using filler_t = std::function<value_t(const Array<float,3>&)>;

  /// Defines a struct which stores information to fill the simulation data.
  struct FillInfo {
    const char* data_name;  //!< Name of the data to fill.
    filler_t    filler;     //!< Callable object to get a value to fill with.
  };

  /// Defines the specification for each of the simulation dimensions.
  struct DimSpec {
    double resolution;
    double size;

    std::size_t elements() const
    {
      return static_cast<std::size_t>(size / resolution);
    }
  };

  /// Defines the type of the container used to store filling information.
  using fillinfo_container_t = std::vector<FillInfo>;

  /// Enables invocation of the derived class constructor through a pointer to
  /// the base class.
  ~Simulator() {}

  /// Runs the simulation.
  virtual void simulate() = 0;

  /// Configures the simulator to set size and resolution of a dimension \p dim.
  /// \param[in] dim  The dimension to specify.
  /// \param[in] spec The specification of the dimension.
  virtual self_t* configure_dimension(std::size_t dim, DimSpec spec) = 0;

  /// Fills the simulator with simulation data for a simulator, where each of
  /// the fillers store the name of the property which they are filling, and 
  /// a function object which can fill values based on their position in the
  /// space.
  /// \param[in] fillers A container of fillers for filling the data.
  virtual void fill_data(fillinfo_container_t&& fillers) = 0;

  /// Prints the results of the simulation to the standard output stream so that
  /// they can be viewed.
  virtual void print_results() const = 0;

  /// Writes the results of the simulation to the \p path using the \p prefix
  /// appended to the property which is output.
  /// \param[in] path   The file path (including the prefix) to write to.
  virtual void write_results(fs::path path) const = 0;

};

}} // namespace fluid::sim

#endif // FLUIDITY_SIMULATOR_SIMULATOR_HPP