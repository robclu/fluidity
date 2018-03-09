//==--- fluidity/algorithm/reduce.cu ----------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  reduce.cu
/// \brief This file defines a kernel to reduce data from a container.
//
//==------------------------------------------------------------------------==//

namespace fluid {

template <typename Iterator, typename Pred, typename...  Args>
__global__ void
reduce(Iterator begin, Iterator end, std::size_t offset, Pred p, Args... args)
{
  p(*begin, *begin[offset], args...);
}

template <typename Iterator, typename Pred, typename... Args>
void reduce_impl(Iterator&& begin, Iterator&& end, Pred&& p, Args&&... args)
{
  
}

} // namespace fluid