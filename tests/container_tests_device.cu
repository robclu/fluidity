//==--- fluidity/tests/container_tests_device.cu ----------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  container_tests_device.cu
/// \brief This file defines tests for device side container functionality.
//
//==------------------------------------------------------------------------==//

#include <fluidity/algorithm/fill.hpp>
#include <fluidity/container/array.hpp>
#include <fluidity/container/device_tensor.hpp>
#include <fluidity/container/host_tensor.hpp>
#include <gtest/gtest.h>

using namespace fluid;
using type_t  = float;
using array_t = Array<type_t, 3>;

TEST(container_device_tensor, can_create_tensor)
{
  device_tensor1d<float> t(20);

  EXPECT_EQ(t.size(), static_cast<decltype(t.size())>(20));
}

/*
TEST(container_host_tensor, can_create_and_initialize_tensor)
{
  const float v = 3.0f;
  device_tensor1d<float> t(20, v);

  auto ht = host_tensor1d<float>(t);
  for (const auto& e : ht)
  {
    EXPECT_EQ(e, v);
  }
}
*/

TEST(container_device_tensor, can_fill_tensor)
{
  device_tensor1d<int> t(20);
  fluid::fill(t.begin(), t.end(), 2);

  auto host_tensor = host_tensor1d<int>(t);
  for (const auto& element : host_tensor)
  {
    EXPECT_EQ(element, 2);
  }
}

struct SetPredicate {
  fluidity_device_only void operator()(int& e)
  {
    e = fluid::flattened_id(fluid::dim_x);
  }
};

TEST(container_device_tensor, can_fill_tensor_with_functor)
{
  device_tensor1d<int> t(20);
  fluid::fill(t.begin(), t.end(), SetPredicate{});

  auto ht    = host_tensor1d<int>(t);
  auto count = 0;
  for (const auto& element : ht) 
  {
    EXPECT_EQ(element, count++);
  }
}

TEST(container_device_tensor, can_resize_tensor)
{
  device_tensor1d<int> t;
  t.resize(30);
  fluid::fill(t.begin(), t.end(), SetPredicate{});

  auto ht    = host_tensor1d<int>(t);
  auto count = 0;
  for (const auto& element : ht) 
  {
    EXPECT_EQ(element, count++);
  }
}

TEST(container_device_tensor, can_get_device_tensor_as_host_tensor)
{
  device_tensor1d<int> t;
  t.resize(30);
  fluid::fill(t.begin(), t.end(), SetPredicate{});

  auto ht    = t.as_host();
  auto count = 0;
  for (const auto& element : ht) 
  {
    EXPECT_EQ(element, count++);
  }
}

fluidity_global void array_multiplication()
{
  constexpr auto value = type_t{2};
  array_t a{value};
  for (const auto& e : a)
  {
    assert(e == value && "Assertation failed!\n");
  }
  auto b = value * a;
  for (const auto& e : b)
  {
    assert(e == value * value && "Assertation failed!\n");
  }  
}

TEST(container_device_array, can_create_and_multiply_device_arrays)
{
  array_multiplication<<<1, 1>>>();
  fluidity_check_cuda_result(cudaDeviceSynchronize());
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
