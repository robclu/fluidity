//==--- fluidity/scheme/upwind.hpp ------------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  upwind.hpp
/// \brief This file defines the implementation of an upwind scheme.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SCHEME_UPWIND_HPP
#define FLUIDITY_SCHEME_UPWIND_HPP

#include "scheme.hpp"

namespace fluid  {
namespace scheme {


struct Upwind : public Scheme<Upwind> {


  template <typename It, typename T, typename... Args>
  fluidity_host_device auto invoke(It&& it, T h, Args&&... args) const
  {

  }
};

}} // namespace fluid::scheme

#endif // FLUIDITY_SCHEME_UPWIND_HPP
#define FLUIDITY_LEVELSET_FIRST_ORDER_EVOLUTION_HPP
