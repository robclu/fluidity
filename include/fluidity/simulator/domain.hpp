//==--- fluidity/simulator/domain.hpp ---------------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  domain.hpp
/// \brief This file defines a simple class to represent a domain.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SIMULATOR_DOMAIN_HPP
#define FLUIDITY_SIMULATOR_DOMAIN_HPP

#include <vector>

namespace fluid {
namespace sim   {

/// Defines a simple class to represent a domain.
struct Domain {
 private:
  /// Defines a struct to hold information for a dimension.
  struct DimInfo {
    double start = 0.0;   //!< The start value of the domension.
    double end   = 1.0;   //!< The end value of the dimension.
  };

  /// Defines the type of the container for the domain's dimension info.
  using dim_info_container_t = std::vector<DimInfo>;

  dim_info_container_t _dim_info;           //!< Info for each dimension.
  double               _resolution = 0.0;   //!< The resolution for the domain.

 public:
  /// Sets the number of dimensions in the domain.
  Domain(std::size_t num_dimensions) : _dim_info(num_dimensions) {};

  /// Sets the number of dimensions in the domain.
  /// \param[in] num_dimensions The number of dimensions in the domain.
  void set_dimensions(std::size_t num_dimensions)
  {
    _dim_info.resize(num_dimensions);
  }

  /// Configures the resolution of the domain.
  /// \param[in] resolution The resolution for the domain.
  void set_resolution(double resolution)
  {
    _resolution = resolution;
  }

  /// Configures the \p dimension to have \p start and \p end values.
  /// \param[in] dimension The dimension to configure.
  /// \param[in] start     The start value of the dimension.
  /// \param[in] end       The end value of the dimension.
  void set_dimension(std::size_t dim, double start, double end)
  {
    _dim_info[dim] = DimInfo{start, end};
  }

  /// Returns the number of elements in a specific \p dim dimension of the
  /// domain for the given \p resolution.
  /// \param[in] dim        The dimension to get the number of elements for.
  std::size_t elements(std::size_t dim) const
  {
    auto& info = _dim_info[dim];
    return static_cast<std::size_t>((info.end - info.start) / _resolution);
  }

  /// Returns the resolution for the domain.
  auto resolution() const
  {
    return _resolution;
  }
};

}} // namespace fluid::sim

#endif // FLUIDITY_SIMULATOR_DOMAIN_HPP