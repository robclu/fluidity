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

#include <fluidity/container/array.hpp>
#include <functional>
#include <vector>

namespace fluid {
namespace sim   {

/// The Simulator class defines the interface for simulation objects.
/// \tparam Traits The traits of the simulator.
//template <typename Traits>
class Simulator {
 public:
  /// Defines the type of this simulator.
  //using self_t   = Simulator<Traits>;
  using self_t   = Simulator;
  /// Defines the traits of the simulation.
  //using traits_t = std::decay_t<Traits>;
  /// Defines the type of the state which is used for the simulation.
  //using state_t  = typename traits_t::state_t;
  /// Defines the type of the data used by the state.
  //using value_t  = typename state_t::value_t;
  /// Defines the type of the functor which can be used to fill elements.
  using filler_t = std::function<float(const Array<float,3>&)>;

  /// Defines the type of the parameter manager used to set the parameters for
  /// the simulation.
  //using param_ptr_t = std::shared_ptr<setting::ParameterManager>;

  /// Defines a struct which stores information to fill the simulation data.
  struct FillInfo {
    const char* data_name;  //!< Name of the data to fill.
    filler_t    filler;     //!< Callable object to get a value to fill with.
  };

  /// Defines the type of the container used to store filling information.
  using fillinfo_container_t = std::vector<FillInfo>;

  /// Enables invocation of the derived class constructor through a pointer to
  /// the base class.
  virtual ~Simulator() {}

  /// Runs the simulation.
  virtual void simulate() = 0;

  //virtual void configure(const setting::Parameter* param)
  //{
  //  std::cout << "Don't know how to set this parameter\n";
  //}

  /// Overload of configuration function to set the CFL number.
  /// \param[in] param The parameter which defins the cfl.
  //virtual void configure(const setting::CflParameter* param) = 0;

  /// Configures the CFL number to use for the simulation.
  /// \param[in] cfl The CFL number for the simulation.
  virtual void configure_cfl(double cfl) = 0;

  /// Configures the simulator to set size and resolution of a dimension \p dim.
  /// \param[in] dim   The dimension to specify.
  /// \param[in] start The start value of the dimension.
  /// \param[in] end   The end value of the dimension.
  virtual void
  configure_dimension(std::size_t dim, double start, double end) = 0;

  /// Configures the simulator to use the \p resolution for the domain.
  /// \param[in] resolution The resolution to use for the domain.
  virtual void configure_resolution(double resolution) = 0;

  /// Configures the simulator to simulate until a certain simulation time.
  /// \param[in] sim_time The time to run the simulation until.
  virtual void configure_sim_time(double sim_time) = 0;

  /// Configures the simulator to simulate for a maximum number of iterations.
  /// \param[in] iters  The maximum number of iterations to simulate for.
  virtual void configure_max_iterations(std::size_t iters) = 0;

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
  virtual void write_results(std::string path) const = 0;

  /// Writes the results of the simulation to the \p path using the \p prefix
  /// appended to the property which is output. This outputs a separate file for
  /// each of the components of the state (i.e, density, pressure, etc), and
  /// outputs the data in the same format as the domain.
  /// \param[in] path   The file path (including the prefix) to write to.
  virtual void write_results_separate_raw(std::string path) const = 0;
};

}} // namespace fluid::sim

#endif // FLUIDITY_SIMULATOR_SIMULATOR_HPP