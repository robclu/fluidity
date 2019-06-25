//==--- fluidity/solver/cuda/material_loader.cuh ----------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  material_loader.cuh
/// \brief This file defines a cuda implementation which loads the material data
///        for each of the materials.
//
//==------------------------------------------------------------------------==//

namespace fluid  {
namespace solver {
namespace detail {
namespace cuda   {

/// This function forwards the material data iterators to the material loader
/// which can use the appropriate method to load the data for the materials.
/// \tparam MaterialLoader    The type of the loader for the material data.
/// \tparam MaterialIterators The iterators for the material data.
template <typename MaterialLoader, typename MaterialIterators>
fluidity_global void load_materials_impl(MaterialIterators&& mat_iterators)
{
  MaterialLoader::invoke(mat_iterators);
}

template <typename Materials>
auto make_material_iterator_wrappers(Materials&& materials)
{
  return unpack(materials, [&] fluidity_host_device (auto&&... mats) 
  {
    return make_tuple(mats.get_iterator_wrapper()...);
  });
}

/// Updater function for updating the simulation using the GPU. This invokes 
/// the updating CUDA kernel for the simulation.
/// 
/// \param[in] solver   The split solver implementation.
/// \param[in] in       The input data to use to update.
/// \param[in] out      The output data to write to after updating.
/// \param[in] dtdh     Scaling factor for the update.
/// \tparam    Solver   The type of the solver.
/// \tparam    It       The type of the mulri dimensional iterator.
/// \tparam    Mat      The type of the material.
/// \tparam    T        The type of the scaling factor.
template <typename MaterialLoader, typename Materials>
void load_materials(MaterialLoader&& loader, Materias&& materials)
{
  using loader_t    = std::decay_t<MaterialLoader>;
  auto iterators    = make_material_iterator_wrappers(materials);
  auto first_it     = get<0>(iterators).state_iterator;
  auto thread_sizes = get_thread_sizes(first_it);
  auto block_sizes  = get_block_sizes(first_it, thread_sizes);

  load_materials_impl<loader_t><<<block_sizes, thread_sizes>>>(iterators);
  fluidity_check_cuda_result(cudaDeviceSynchronize());
}

}}}} // namespace fluid::solver::detail::cuda