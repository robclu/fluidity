//==--- fluidity/container/storage_traits.hpp -------------- -*- C++ -*- ---==//
//
//                                Fluidity
//            
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  storage_traits.hpp
/// \brief This file defines functionality for specifying the traits related to
///        storage for containers.
//
//==------------------------------------------------------------------------==//

#include <fluidity/utility/portability.hpp>

#ifndef FLUIDITY_CONTAINER_STORAGE_TRAITS_HPP
#define FLUIDITY_CONTAINER_STORAGE_TRAITS_HPP

namespace fluid  {
namespace traits {

/// The StorageFormat enum class defines the options for the storage format
/// which can be used to store data.
enum class StorageFormat : uint8_t {
  array_of_struct_t = 0,    //!< Stores the data in AOS.
  struct_of_array_t = 1     //!< Stores the data as SOA.
};


/// This class does not provide any functionality other than to allow the user
/// to specify that a class derived from it has a N elements of the same type
/// and size, and can therefore be stored as SoA or as AoS. Which allows for
/// significant performance increased where it is more computationally efficient
/// to compute on SoA data.
///
/// For example:
///
/// \code{cpp}
/// template <typename T>
/// struct Vec3 : SingleType {
///   T x, y, z;  
/// };
/// \endcode
///
/// Can then be stored as SoA or AoS in the custom array classes.
struct SingleType {};

/// Returns true if the class T has only a single data type and can therefore
/// has flexible storage in that it can be SoA or SoA.
/// \tparam T The type to check if the storage is configurable.
template <typename T>
static constexpr bool flexible_storage_v = std::is_base_of_v<SingleType, T>;

}} // namespace fluid::traits

#endif // FLUIDITY_CONTAINER_STORAGE_TRAITS_HPP

