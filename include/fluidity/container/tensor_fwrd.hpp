//==--- fluidity/container/tensor_fwrd.hpp ----------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  tensor_fwrd.hpp
/// \brief This file forward declares Tensor classes.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_CONTAINER_TENSOR_FWRD_HPP
#define FLUIDITY_CONTAINER_TENSOR_FWRD_HPP

namespace fluid {

/// Declaration of a device tensor class which specializes the components of
/// the tensor implementation which are specific to the device side.
/// \tparam T          The type of the data stored in the tensor.
/// \tparam Dimensions The number of dimensions in the tensor.
template <typename T, std::size_t Dimensions>
class DeviceTensor;  

/// Declaration of a host tensor class which specializes the components of
/// the tensor implementation which are specific to the host side.
/// \tparam T          The type of the data stored in the tensor.
/// \tparam Dimensions The number of dimensions in the tensor.
template <typename T, std::size_t Dimensions>
class HostTensor;

} // namespace fluid

#endif // FLUIDITY_CONTAINER_TENSOR_FWRD_HPP