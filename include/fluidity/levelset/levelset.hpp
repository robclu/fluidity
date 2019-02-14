//==--- fluidity/levelset/levelset.hpp --------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  levelset.hpp
/// \brief This file defines level set functionality.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_LEVELSET_LEVELSET_HPP
#define FLUIDITY_LEVELSET_LEVELSET_HPP

#include <fluidity/algorithm/fill.hpp>
#include <fluidity/algorithm/for_each.hpp>
#include <fluidity/container/device_tensor.hpp>
#include <fluidity/container/host_tensor.hpp>
#include <fluidity/execution/execution_policy.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid {

/// The LevelSet class 
template <typename         T                              , 
          std::size_t      Dims                           ,
          exec::DeviceKind Kind    = exec::DeviceKind::gpu,
          typename         Storage =
            std::conditional_t<
              Kind == exec::DeviceKind::gpu,
                DeviceTensor<T, Dims>, HostTensor<T, Dims>>>
class LevelSet {
  /// Defines the data type used for the level set.
  using value_t   = std::decay_t<T>;
  /// Defines the type of the storage.
  using storage_t = Storage;

  /// Enabling function to check that a size is specified for each dimension.
  /// \tparam Sizes The sizes for each dimension.
  template <typename... Sizes>
  using size_enable_t = std::enable_if_t<sizeof...(Sizes) == Dims, int>;

  /// Defines the number of dimensions for level set.
  static constexpr auto num_dimensions = Dims;

  storage_t _data; //!< Data for the level set.
 public:
  /// Constructor to initialize the level set data with a predicate and the
  /// sizes of the dimensions for the level set.
  template <typename    Pred           ,
            typename... Sizes          ,
            size_enable_t<Sizes...> = 0>
  LevelSet(Pred&& pred, Sizes&&... sizes)
  : _data(std::forward<Sizes>(sizes)...)
  {
    fill(_data.multi_iterator(), [&] fluidity_host_device (auto it)
    {
      auto positions = Array<float, num_dimensions>{};
      unrolled_for<num_dimensions>([&] (auto dim)
      {
        positions[dim] = static_cast<float>(flattened_id(dim)) 
                       / static_cast<float>(it.size(dim));
      });
      pred(it, positions);
    });
  }

  void print() const
  {
    auto data = _data.as_host();
    auto it   = data.multi_iterator();

    if (num_dimensions == 1)
    {
      for (int i : range(it.size(0)))
      {
        std::cout << *it << " ";
        it.shift(1, std::size_t{0});
      }
    }
    std::cout << "\n";
    return;

    if (num_dimensions == 2)
    {
      for (const auto j : range(it.size(1)))
      {
        it.shift(1, std::size_t{1});
        for (const auto i : range(it.size(0)))
        {
          std::cout << *it.offset(i, std::size_t{0}) << " ";
        }
        std::cout << "\n";
      }
    }
    std::cout << "\n";  
    return;
  }
};

} // namespace fluid

#endif // FLUIDITY_LEVELSET_LEVELSET_HPP