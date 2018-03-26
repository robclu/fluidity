//==--- fluidity/tests/algorithm_tests_device.cu ----------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  algorithm_tests_device.cu
/// \brief This file defines tests for the algorithm module to ensure that they
///        work correctly on the device side.
//
//==------------------------------------------------------------------------==//

#include <gtest/gtest.h>
#include <fluidity/algorithm/algorithm.hpp>
#include <fluidity/container/device_tensor.hpp>
#include <fluidity/container/host_tensor.hpp>

// Alias for the type of host tensor used for the tests.
template <typename T>
using host_tensor1d   = fluid::HostTensor<T, 1>;

// Alias for the type of device tensor used for the tests.
template <typename T>
using device_tensor1d = fluid::HostTensor<T, 1>;

using element_t = int;

struct SumPredicate {
  fluidity_host_device void operator()(element_t& a, const element_t& b)
  {
    a += b;
  }
};

TEST(algorithm_device_tests, can_reduce_container)
{
  const auto size  = 20;
  const int  value = 2;
  device_tensor1d<element_t> t(size);
  fluid::fill(t.begin(), t.end(), value);

  auto result = fluid::reduce(t.begin(), t.end(), SumPredicate{});
  EXPECT_EQ(result,  value * size);
}


struct SetPredicate {
  fluidity_host_device void operator()(element_t& a, const element_t& /*b*/)
  {
    a = fluid::flattened_id(fluid::dim_x);
  }
};

TEST(algorithm_device_tests, can_get_max_element)
{
  const auto size = 20;
  device_tensor1d<element_t> t(size);

  fluid::fill(t.begin(), t.end(), SetPredicate{});
  auto result = fluid::max_element(t.begin(), t.end());

  EXPECT_EQ(result, size - 1);
}

TEST(algorithm_device_tests, can_get_min_element)
{
  const auto size = 20;
  device_tensor1d<int> t(size);

  fluid::fill(t.begin(), t.end(), SetPredicate{});
  auto result = fluid::min_element(t.begin(), t.end());

  EXPECT_EQ(result, 0);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
