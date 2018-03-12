//==--- fluidity/algorithm/reduce.hpp ---------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  reduce.hpp
/// \brief This file defines the implementation of reduction of a container.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_ALGORITHM_REDUCE_HPP
#define FLUIDITY_ALGORITHM_REDUCE_HPP

namespace fluid {

/// Reduces the data pointed to by the \p begin and \p end iterator.
/// 
/// ~~~cpp
/// auto result = reduce(data.begin(), data.end(),
///   [] fluidity_host_device (auto&& left, auto&& right)
///   {
///     left += right;
///   }
/// );
/// ~~~
/// 
/// The algorithm will handle any number of elements (incuding random odd sizes)
/// and will also reduce a multidimensional container (see the test). When the
/// data has an odd number of elements (or when an odd number of elements
/// arises -- for example reducing 122 elements will become 61 on the next
/// iteration) the first and last floor(n/2) elements are reduced __missing__
/// the middle element (in the case of 61 elements, [0-29] reduce with [31-60] 
/// and 30 is skipped) and the size is divided by two (and floored) and then the
/// last bit of the size is added back in to the size (which only does anything
/// when the size is odd, so for 61 elements the size modification is 61 / 2 +
/// (61 & 0x1)) which will include the skipped element in the next iteration.
/// This also uses no modulus or type conversion, and only adds log2 bitwise
/// ands and additions. The benchmarks show good performance.
/// 
/// \note Predicates must use ``fluidity_host_device`` so that they can execute
///       on the GPU.
///       
/// \todo   Change __syncthreads() to sync() to support CPU and GPU block sync.
///
/// \param[in]  begin     An iterator to the beginning of the data.
/// \param[in]  end       An iterator to the end of the data.
/// \param[in]  pred      The predicate to apply to element pairs.
/// \param[in]  args      Additional arguments for the predicate.
/// \tparam     Iterator  The type of the iterator.
/// \tparam     Pred      The type of the predicate.
/// \tparam     Args      The type of any additional args for the predicate.
template <typename Iterator, typename Pred, typename... Args>
fluidity_host_device decltype(auto)
reduce(Iterator&& begin, Iterator&& end, Pred&& pred, Args&&... args)
{
  reduce_impl(begin, end, pred, args...);
}

} // namespace fluid

#endif // FLUIDITY_ALGORITHM_REDUCE_HPP