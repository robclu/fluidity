//==--- fluidity/tests/container_tests_host.cpp ------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  container_tests_host.cpp
/// \brief This file defines tests for host side container functionality.
//
//==------------------------------------------------------------------------==//

#include <fluidity/container/host_tensor.hpp>
#include <gtest/gtest.h>

template <typename T>
using host_tensor1d = fluid::HostTensor<T, 1>;

TEST(container_host_tensor, can_create_tensor) {
  host_tensor1d<float> t(20);

  EXPECT_EQ(t.size(), static_cast<decltype(t.size())>(20));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
