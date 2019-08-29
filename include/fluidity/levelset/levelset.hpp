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

namespace fluid    {
namespace levelset {

/// The LevelSet class 
template <
  typename         T                              , 
  std::size_t      Dims                           ,
  exec::DeviceKind Kind    = exec::DeviceKind::gpu,
  typename         Storage =
    std::conditional_t<
      Kind == exec::DeviceKind::gpu,
      DeviceTensor<T, Dims>        ,
      HostTensor<T, Dims>
    >
>
class LevelSet {
  /// Defines the data type used for the level set.
  using value_t   = std::decay_t<T>;
  /// Defines the type of the storage.
  using storage_t = Storage;
  /// Defines the type of the pointer used by the storage.
  using pointer_t = typename storage_t::pointer_t;

  /// Enabling function to check that a size is specified for each dimension.
  /// \tparam Sizes The sizes for each dimension.
  template <typename... Sizes>
  using size_enable_t = std::enable_if_t<sizeof...(Sizes) == Dims, int>;

  /// Defines the number of dimensions for level set.
  static constexpr auto num_dimensions = Dims;

  storage_t _data; //!< Data for the level set.
 public:
  /// Default constructor for the levelset.
  LevelSet() = default;

  /// Constructor to initialize the level set data with a predicate and the
  /// sizes of the dimensions for the level set.
  template <typename    Pred           ,
            typename... Sizes          ,
            size_enable_t<Sizes...> = 0>
  LevelSet(Pred&& pred, Sizes&&... sizes)
  : _data(std::forward<Sizes>(sizes)...) {
    fill(_data.multi_iterator(), [&] fluidity_host_device (auto it) {
      auto positions = Array<float, num_dimensions>{};
      unrolled_for<num_dimensions>([&] (auto dim) {
        positions[dim] = static_cast<float>(flattened_id(dim)) 
                       / static_cast<float>(it.size(dim));
      });
      pred(it, positions);
    });
  }

  /// Resizes the level set data.
  template <typename... Sizes, size_enable_t<Sizes...> = 0>
  void resize(Sizes&&... sizes) {
    _data.resize(std::forward<Sizes>(sizes)...);
  }

  /// Resizes a single dimension \p dim to have \p elements number of elements.
  /// \param[in] dim      The dimension to resize.
  /// \param[in] elements The number of elements to resize the dimension to.
  void resize_dim(std::size_t dim, std::size_t elements) {
    _data.resize_dim(dim, elements);
  }

  /// Initializes the levelset data using the \p predicate.
  template <typename Pred>
  void initialize(Pred&& pred) {
    fill(_data.multi_iterator(), [&] fluidity_host_device (auto it) {
      auto positions = Array<float, num_dimensions>{};
      unrolled_for<num_dimensions>([&] (auto dim)
      {
        positions[dim] = static_cast<float>(flattened_id(dim)) 
                       / static_cast<float>(it.size(dim));
      });
      pred(it, positions);
    });    
  }

  /// Returns the storage for the levelset for the host.
  auto host_storage() const {
    return std::move(_data.as_host());
  }

  /// Returns a multi-dimensional iterator over the levelset data.
  auto multi_iterator() const {
    return _data.multi_iterator();
  }

  /// Resets the levelset data.
  fluidity_host_device void reset_data(pointer_t new_data) {
    _data.reset_data(new_data);
  }

  void print() const
  {
    auto data = _data.as_host();
    auto it   = data.multi_iterator();

    if (num_dimensions == 1)
    {
      for (int i : range(it.size(0)))
      {
        std::cout
          << std::setfill(' ') << std::setw(6) << std::left 
          << i<< " "
          << std::setfill(' ') << std::setw(8) << std::right
          << std::setprecision(4) << *it
          << "\n";
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

///==--- Functions ---------------------------------------------------------==//

/// Returns true if dereferencing the \p levelset_it has a negative value (i.e
/// the data value is inside the levelset), or is equal to zero (i.e on the
/// boundary). 
/// \param[in] levelset_it An iterator over levelset data.
/// \tparam    LSIT        The type of the levelset iterator.
template <typename LSIT>
fluidity_host_device constexpr auto inside(LSIT&& levelset_it) -> bool {
  using type_t = std::decay_t<decltype(*levelset_it)>;
  return *levelset_it <= type_t{0} || *levelset_it <= -type_t{0};
}

/// Returns true if dereferencing the \p levelset_it has a positive value (i.e
/// the data value is outside the levelset).
/// \param[in] levelset_it An iterator over levelset data.
/// \tparam    LSIT        The type of the levelset iterator.
template <typename LSIT>
fluidity_host_device constexpr auto outside(LSIT&& levelset_it) -> bool {
  using type_t = std::decay_t<decltype(*levelset_it)>;
  return *levelset_it > type_t{0};
}

/// Returns true if dereferencing the \p levelset_it has a value of zero (i.e
/// the data value is on the boundary).
/// \param[in] levelset_it An iterator over levelset data.
/// \tparam    LSIT        The type of the levelset iterator.
template <typename LSIT>
fluidity_host_device constexpr auto on_boundary(LSIT&& levelset_it) -> bool {
  using type_t = std::decay_t<decltype(*levelset_it)>;
  return *levelset_it == type_t{0} || *levelset_it == -type_t{0};
}

/// Returns true if dereferencing the \p levelset_it has a value less than 0
/// (i.e it's inside the levelset), but larger than the negative of the
/// resolution, so the cell is an interfacial cell inside the levelset.
/// \param[in] levelset_it An iterator over levelset data.
/// \param[in] dh          The resolution for the levelset grid.
/// \tparam    LSIT        The type of the levelset iterator.
/// \tparam    T           The type of the resolution data.
template <typename LSIT, typename T>
fluidity_host_device constexpr auto
inside_interfacial_cell(LSIT&& levelset_it, T dh) -> bool {
  return (inside(levelset_it) || on_boundary(levelset_it)) && 
         (*levelset_it >= -dh);
}

/// Returns true if dereferencing the \p levelset_it has a value greater than 0
/// (i.e it's outside the levelset), but less than the resolution, so the cell
/// is an interfacial cell outside the levelset.
/// \param[in] levelset_it An iterator over levelset data.
/// \param[in] dh          The resolution for the levelset grid.
/// \tparam    LSIT        The type of the levelset iterator.
/// \tparam    T           The type of the resolution data.
template <typename LSIT, typename T>
fluidity_host_device constexpr auto
outside_interfacial_cell(LSIT&& levelset_it, T dh) -> bool {
  return outside(levelset_it) && (*levelset_it < std::abs(dh));
}

}} // namespace fluid::levelset

#endif // FLUIDITY_LEVELSET_LEVELSET_HPP
