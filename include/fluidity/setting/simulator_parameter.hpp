//==--- fluidity/setting/simulator_parameter.hpp ----------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  simulator_parameter.hpp
/// \brief This file defines a class which implements the Parameter interface
///        to allow paramters for the simulator to be set.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_SIMULATOR_PARAMETER_HPP
#define FLUIDITY_SETTING_SIMULATOR_PARAMETER_HPP

#include "parameter.hpp"
#include "settings.hpp"
#include <fluidity/utility/string.hpp>

namespace fluid   {
namespace setting {

/// The SimualtorParameter class allows parameters of the simulator to be
/// specified.
struct SimulatorParameter : public Parameter {
 private:
  /// Defines the format for a domain parameter.
  static constexpr const char* format = 
    "simulator : {\n"
    "  [data_type     : value][,]\n"
    "  [limit_form    : value][,]\n"
    "  [limiter       : value][,]\n"
    "  [reconstructor : value][,]\n"
    "  [flux_method   : value][,]\n"
    "  [solve_method  : value][,]\n"
    "  [execution     : value][,]\n"
    "}\n";

 public:
  /// Defines the type of the container for the dimension information.
  using container_t = std::vector<Setting>;

  /// Returns the string identifier which defines the name of the setting.
  std::string type() const final override
  {
    return "simulator";
  }

  /// Returns a vector of identifiers for the names of settings for the domain.
  std::vector<std::string> param_options() const final override
  {
    return std::vector<std::string>{
      "data_type"    ,
      "limit_form"   ,
      "limiter"      ,
      "material"     ,
      "reconstructor",
      "flux_method"  ,
      "solve_method" ,
      "execution"
    };
  }

  /// Returns a string which defines the format for the parameter.
  std::string format_info() const final override
  {
    return format;
  }

  /// Returns a string with the information for the parameter.
  std::string display_string() const final override
  {
    std::ostringstream stream;
    util::format_name_value(stream, "name", type(), 0, 4);
    for (const auto& setting : _settings)
    {
      util::format_name_value(stream, setting.name, setting.value, 2, 13);
    }
    return stream.str();
  }

  /// Sets the dimension information for a dimension.
  /// \param[in] sim_setting The setting whose value contains the information
  ///                        to use to set the simulator parameters.
  bool try_set(const Setting& sim_setting) final override
  {
    auto settings = Settings::from_string(sim_setting.value);
    for (auto& setting : settings)
    {
      if (setting.complex || !this->valid_param(setting.name))
      {
        std::cout << "Failed to parse simulator setting : " << setting.name
                  << ". Simulator settings must all be { name : value }, with "
                  << "format:\n" << format;
          return false;
      }
      util::remove(setting.value, ',', '}','\n');
      _settings.push_back(setting);
    }
    return true;
  }

 private:
  container_t _settings; //!< Information for the siulator.
};

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_SIMULATOR_PARAMETER_HPP