//==--- fluidity/setting/option_manager.hpp ---------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  option_manager.hpp
/// \brief This file defines a class to manager different compile time options
///        which can be set at runtime.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_OPTION_MANAGER_HPP
#define FLUIDITY_SETTING_OPTION_MANAGER_HPP

#include "option.hpp"
#include "option_tuple.hpp"
#include <fluidity/algorithm/for_each.hpp>
//#include "parameter_manager.hpp"

namespace fluid   {
namespace setting {

/// The OptionManager class allows a set of options to be defined, where each
/// option conforms to the Option interface. Each option has a ```type``` string
/// which defines the string which can be used to set an option of that type at 
/// runtime. The Option also holds a list of OptionHolder template types, where
/// each OptionHolder holds the type to create as a template and a string which
/// defines the runtime value which can be used to select that type.
/// 
/// The OptionManager is templated on a list of the Option types, and should be
/// composed inside a templated Derived class, and can then be used to create a
/// unique pointer to the base class through a templated Derived class, where
/// the options for the template paramters of the derived class are defined by
/// the OptionHolder types in the Options of the OptionManager.
///
/// The only way to provide this functionality is to create function overloads
/// for all combinations of the template options in the OptionHolders, which
/// can take a while to compile if there are many options. To allow for faster
/// compile times, a ``create_default<>()`` method is provided which uses
/// the default template options in the Derived class to reduce compile times
/// for debug builds. For release builds, the compile time is not of concern and
/// the additional performance (and more especially the lack of having to use
/// virtual methods for implementations) is extremely beneficial, especially for
/// GPU code.
///
/// The option template paramters must all be derived from the Option class,
/// which is asserted during construction. The Option implementation define the
/// ```type``` string and a list of possible values as strings, which can then
/// be set through the OptionManager using the ```set(op_type, op_value)```
/// method.
/// 
/// An example use case for creating a Simulator object, which is configurable
/// (the flux method and material are compile time options) through the
/// template parameters in the derived MultimaterialSimulator class, is the following
/// (additionally, for a full example, see the setting_test.cpp tests):
///
/// \begin{code}
/// // Define the option classes:
/// struct FluxOption : Option<FluxOption> {
///   static constexpr const char* type = "flux";
///   constexpr auto choice_list() {
///     return std::make_tuple(
///       OptionHolder<Hllc>("hllc"), 
///       OptionHolder<Force>("force")
///     );
///   }
///   ...
/// };
///
/// struct MaterialOption : Option<MaterialOption> {
///    ...
/// }
///
/// // Define the base object to create:
/// struct Simulator {
///   virtual void run() = 0;
/// };
///
/// // Define the templated implementation:
/// template <typename... Ts>
/// struct MultimaterialSimulator : Simulator {
///   // Required to be able to build:
///   template <typename... Us>
///   using make_type_t = MultimaterialSimulator<Us...>;
///
///   // Define which options are at which positions, and the default values:
///   using flux_t     = type_at_t<0, Hllc, Ts...>;
///   using material_t = type_at_t<1, Ideal, Ts...>;
///
///   // Define the option manager, matching options to above positions:
///   using option_manager_t = 
///     OptionManager<Simulator, MultimaterialSimulator<>, FluxOption, MaterialOption>;
///
///   virtual void run() override { ... }
/// };
///
/// // Aliases for the simulator:
/// using simulator_t = MultimaterialSimulator<>;
///
/// /// Alias for the option manager.
/// using op_manager_t = typename simulator_t::option_manager_t;
///
/// int main()
/// {
///   op_manager_t op_manager;
///
///   // Configure the types:  
///   op_manager.set("flux", "force").set("material", "nonideal");
///
///   // Get the simulation object (using the set options:):
///   auto sim = op_manager.create();
///
///   // If wanting to use the default types (will use Hllc and Ideal):
///   auto sim = op_manager.create_default();
/// }
/// \end{code}
/// \tparam Base    The type of the base class for creation.
/// \tparam Derived The type of the derived class to use to create the Base.
/// \tparam Options The list of Option types.
template <typename    Base        ,
          typename    Derived     ,
          typename    ParamManager,
          typename... Options     >
struct OptionManager
{
  /// Defines the type of the base pointer to create.
  using base_t          = Base;
  /// Defines the type of the derived pointer to use to create the base pointer.
  using derived_t       = Derived;
  /// Defines the type of the options.
  using options_t       = OptionTuple<Options...>;
  /// Defines the type of the parameter manager
  using param_manager_t = std::shared_ptr<ParamManager>;
  //using param_manager_t = std::shared_ptr<ParameterManager>;
    
  /// Defines the total number of options.
  static constexpr auto total_opts_v    = sizeof...(Options);
  /// Defines the string for all options.
  static constexpr const char* all_opts = "all";

  /// Constructor which checks that each of the options conforms to Option
  /// interface.  
  OptionManager()
  {
    _param_manager = std::make_shared<ParamManager>();
  }

  /// Displays the option informtation for the option \p op_type, or for all 
  /// options by default.
  /// \param[in] op_type The option to display the information for.
  void display_option_info(std::string op_type = all_opts)
  {
    for_each_option(this->_opts, [&] (auto& opt)
    {
      if (opt.is_valid_option_type(op_type) || std::string(all_opts) == op_type)
      {
        opt.print();
      }
    });
  }

  /// Sets the parameter manager for the option manager.
  /// \param  manager The parameter manager.
  /// \tparam Manager The type of the parameter manager implementation.
  template <typename Manager>
  void set_param_manager(std::unique_ptr<Manager>&& manager)
  {
    _param_manager = std::move(manager);
  }

  /// Configures the option manager from a Settings configuration.
  /// \param[in] settings The settings to configure the manager with.
  void configure(const std::vector<Setting>& settings)
  {
    for (auto& setting : settings)
    {
      /*
      if (setting.used)
      {
        continue;
      }
      */
      //if (try_set_option(setting.name, setting.value))
      //{
        //setting.used = true;
      //  continue;
        //std::cout << "Setting set:\n" << setting << "\n";
      //}
      if (_param_manager->try_set_param(setting))
      {
        continue;
      }
      /*
      std::cout << "\tUnused setting:\n" << setting << "\n"
                << "\tSetting does not match any simulator option type "
                << "or simulation paramter type.\n";
      */
    }
    _param_manager->check_parameters_are_set();
    _param_manager->display_parameters();
  }

  /// Tries to sets an option with type \p op_type to have the value defined by
  /// \p op_value, returning true on success and false on failure.
  /// \param[in] op_type  The type of the option.
  /// \param[in] op_value The value of the option to set.
  bool try_set_option(const std::string& op_type, const std::string& op_value)
  {
    bool set = false;
    for_each_option(this->_opts, [&] (auto& opt)
    {
      if (opt.is_valid_option_type(op_type))
      {
        if (opt.set_option_value(op_value))
        {
          set = true;
        }
      }
    });
    return set;
  }

  /// Tries to set a simulation parameter with type \p param_type to have the
  /// values(s) defined by the sequence \p seq, returning true on success and
  /// false on failure.
  /// \param[in] param_type The type of the parameter
  /// \param[in] seq        The sequence to get the values from.
  bool try_set_param(const Setting& setting)
  {
    return _param_manager->try_set_param(setting);
  }

  /// Creates a unique pointer to a Base class type using a Derived type, where
  /// the Derived type is a class template and the template paramters are filled
  /// by the runtime configured options.
  auto create() const
  {
    std::unique_ptr<base_t> base = std::make_unique<derived_t>();
    _opts.template create<derived_t>(base);
    return std::move(base);
  }

  /// Creates a unique pointer to a Base class type using a Derived type. This
  /// uses the default types defines in the Derived class and can be used during
  /// debugging to improve compile times.
  auto create_default() const
  {
    /* namespace lg = logging;
       lg::log(lg::logger_t, 
               "Using default options for option manager, "
               "and options set using .set(op_type, op_value) will not be "
               "applied. Please use create_from_options<>() to use set "
               "options. This method is a debug build optimisation.");
    */
    std::unique_ptr<base_t> base = std::make_unique<derived_t>();
// /    base->configure(_param_manager);
    for_each(_param_manager->get_params(), [&] (const auto& param)
    {
      //base->configure(param);
    });
    return std::move(base);
  }

   /// Creates a unique pointer to a Base class type using a Derived type. This
  /// uses the default types defines in the Derived class and can be used during
  /// debugging to improve compile times.
  auto create_simulator() const
  {
    /* namespace lg = logging;
       lg::log(lg::logger_t, 
               "Using default options for option manager, "
               "and options set using .set(op_type, op_value) will not be "
               "applied. Please use create_from_options<>() to use set "
               "options. This method is a debug build optimisation.");
    */
    auto sim = std::make_unique<derived_t>();
// /    base->configure(_param_manager);
    for_each(_param_manager->get_params(), [&] (const auto& param)
    {
      //base->configure(param);
    });
    return std::move(sim);
  }

 private:
  options_t       _opts;          //!< The options to manage.
  param_manager_t _param_manager; //!< The manager for simulation parameters.
};

}} // fluid::setting

#endif // FLUIDITY_SETTING_OPTION_MANAGER_HPP