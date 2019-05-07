//==--- fluidity/material/combine_materials.hpp ------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2019 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  combine_materials.hpp
/// \brief This file combines the data for multiple materials into a single
///        material.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_MATERIAL_COMBINE_MATERIALS_HPP
#define FLUIDITY_MATERIAL_COMBINE_MATERIALS_HPP

#include "cuda/combine_materials.cuh"

namespace fluid    {
namespace material {
namespace detail   {

/// Implementation of the material combining functionality for the case that the 
/// iterators have a CPU execution policy, which simply forwards the iterators
/// to the parallel CPU implementation. This overload is only enabled when the
/// iterators have a CPU execution policy.
/// \param[in] iterators         The iterators over the material data.
/// \tparam    MaterialIterators The type of the iterator container.
template <typename MaterialIterators, exec::cpu_enable_t<MaterialIterators> = 0>
void combine_into_first_impl(MaterialIterators&& iterators)
{
  // TODO: Add parallel CPU implementation.
}

/// Implementation of the material combining functionality for the case that the 
/// iterators have a GPU execution policy, which simply forwards the iterators
/// to the CUDA kernel.  This overload is only enabled when the iterators have a
/// GPU execution policy.
/// \param[in] iterators         The iterators over the material data.
/// \tparam    MaterialIterators The type of the iterator container.
template <typename MaterialIterators, exec::gpu_enable_t<MaterialIterators> = 0>
void combine_into_first_impl(MaterialIterators&& iterators)
{
  cuda::combine_into_first(std::forward<MaterialIterators>(iterators));
}

} // namespace detail

/// Interface for combining the data from all the materials into the first
/// material in the container of materials such that the state data for the
/// first material is correct for the entire domain. This is useful at the end
/// of simulations so that the data can be visualized.
/// \param[in] materials        The materials which hold the data.
/// \tparam    MaterialContainer The container for the materials.
template <typename MaterialContainer>
void combine_into_first(MaterialContainer&& materials)
{
  detail::combine_into_first_impl(
    unpack(materials, [&] (auto&&... mat_data)
    {
      return make_tuple(mat_data.material_iterator()...);
    })
  );
}


}} // namespace fluid::material

#endif // FLUIDITY_MATERIAL_COMBINE_MATERIALS_HPP
