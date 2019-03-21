//==--- fluidity/solver/material_loader.hpp ---------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  material_loader.hpp
/// \brief This file defines an implementation for a material loader.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SOLVER_MATERIAL_LOADER_HPP
#define FLUIDITY_SOLVER_MATERIAL_LOADER_HPP

#include <fluidity/utility/cuda.hpp>
#include <fluidity/utility/portability.hpp>

namespace fluid  {
namespace solver {

/// The MaterialLoader struct loads data for all materials in a material tuple,
/// such that each material has valid data inside it's levelset as well as in
/// the cells outside the material levelset which are required to evolve the
/// material.
struct MaterialLoader {
  template <typename MaterialIterators>
  void invoke(MaterialIterators&& iterators)
  {
    printf("Loading materials ...");
  }
};

}} // namespace fluid::solver

#endif // FLUIDITY_SOLVER_MATERIAL_LOADER_HPP