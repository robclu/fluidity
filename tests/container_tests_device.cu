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
#include <fluidity/container/device_tensor.hpp>
#include <fluidity/container/host_tensor.hpp>
#include <gtest/gtest.h>

template <typename T>
using device_tensor1d = fluid::DeviceTensor<T, 1>;

template <typename T>
using host_tensor1d = fluid::HostTensor<T, 1>;

TEST(container_device_tensor, can_create_tensor)
{
  device_tensor1d<float> t(20);

  EXPECT_EQ(t.size(), static_cast<decltype(t.size())>(20));
}

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

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
