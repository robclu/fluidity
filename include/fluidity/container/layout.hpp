//==--- fluidity/container/layout.hpp ---------------------- -*- C++ -*- ---==//
//
//                                Fluidity
//            
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  layout.hpp
/// \brief This file defines a structure which allows the definition of storage
///        layout for contiguous data structures.
//
//==------------------------------------------------------------------------==//

#include <fluidity/utility/portability.hpp>

namespace fluid {

/// The StorageFormat enum class defines the options for the storage format
/// which can be used to store data.
enum class StorageFormat : uint8_t {
  array_of_struct_t = 0,    //!< Stores the data in AOS.
  struct_of_array_t = 1     //!< Stores the data as SOA.
};

/// Defines a struct which specifies the traits of the layout for an array.
/// \tparam T       The type to get the layout traits for.
/// \tparam Format  The desired storage format for the layout.
template <typename T, StorageFormat PreferredStorage>
struct LayoutTraits {

 private:
  /// Checks if the type to get the layout traits for is a base class of
  /// SingleType and can therefore be configured for SOA format.
  //static constexpr configureable_storage_v = std::is_base_of_v<>;
};

/// Defines an array class.
template <typename    T       ,
          std::size_t Elements,
          typename    Format  = StorageFormat::array_of_struct_t>
class Array {

};

} // namespace fluid