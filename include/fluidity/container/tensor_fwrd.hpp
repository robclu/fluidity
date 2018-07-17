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
/// 
/// This will work on all the algorithms in the fluid namespace as the
/// HostTensor would. However, it will not work with a range-based-for on the
/// host side since the iterator points to a device-side element. To use the
/// tensor with a range-based-for on the host side the conversion to host must
/// be used:
/// 
/// ~~~{.cpp}
/// auto dev_tensor = DeviceTensor<int, 1>(50);
/// for (const auto& e : dev_tensor.as_host())
/// {
///   // Do something ...
/// }
/// ~~~
/// 
/// To convert to the equivalent version of the tensor on the host-side, use
/// the ``as_host()`` function, and using the ``as_device()`` function will
/// return a copy of the tensor.
/// 
/// ~~~{.cpp}
/// auto dev_tensor = DeviceTensor<int, 1>(50);
/// 
/// // Get a host version:
/// auto host_tensor = dev_tensor.as_host();
/// 
/// // Get a new device version:
/// auto dev_tensor1 = dev_tensor.as_device();
/// auto dev_tensor2 = host_tensor.as_device();
/// ~~~
/// 
/// \tparam T          The type of the data stored in the tensor.
/// \tparam Dimensions The number of dimensions in the tensor.
template <typename T, std::size_t Dimensions>
class DeviceTensor;  

/// Declaration of a host tensor class which specializes the components of
/// the tensor implementation which are specific to the host side.
/// 
/// To convert to the equivalent version of the tensor on the device-side, use
/// the ``as_device()`` function, and using the ``as_host()`` function will
/// return a copy of the tensor.
/// 
/// ~~~{.cpp}
/// auto host_tensor = HostTensor<int, 1>(50);
/// 
/// // Get a device version:
/// auto dev_tensor = host_tensor.as_device();
/// 
/// // Get a new host version:
/// auto host_tensor1 = dev_tensor.as_host();
/// auto host_tensor2 = host_tensor.as_host();
/// ~~~
/// 
/// \tparam T          The type of the data stored in the tensor.
/// \tparam Dimensions The number of dimensions in the tensor.
template <typename T, std::size_t Dimensions>
class HostTensor;

//==--- ALiases ------------------------------------------------------------==//

/// Alias for a 1-dimensional host side tensor.
/// \tparam T The type of the data for the tensor.
template <typename T>
using host_tensor_1d_t = HostTensor<T, 1>;

/// Alias for a 1-dimensional device side tensor.
/// \tparam T The type of the data for the tensor.
template <typename T>
using device_tensor_1d_t = DeviceTensor<T, 1>;

/// Alias for a 2-dimensional host side tensor.
/// \tparam T The type of the data for the tensor.
template <typename T>
using host_tensor_2d_t = HostTensor<T, 2>;

/// Alias for a 2-dimensional device side tensor.
/// \tparam T The type of the data for the tensor.
template <typename T>
using device_tensor_2d_t = DeviceTensor<T, 2>;

/// Alias for a 3-dimensional host side tensor.
/// \tparam T The type of the data for the tensor.
template <typename T>
using host_tensor_3d_t = HostTensor<T, 3>;

/// Alias for a 3-dimensional device side tensor.
/// \tparam T The type of the data for the tensor.
template <typename T>
using device_tensor_3d_t = DeviceTensor<T, 3>;

} // namespace fluid

#endif // FLUIDITY_CONTAINER_TENSOR_FWRD_HPP