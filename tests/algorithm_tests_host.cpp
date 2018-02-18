//==--- fluidity/tests/algorithm_tests_host.cpp ------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  algorithm_tests_host.cpp
/// \brief This file defines tests for the algorithm module to ensure that they
///        work correctly on the host side.
//
//==------------------------------------------------------------------------==//

#include <gtest/gtest.h>
#include <fluidity/algorithm/algorithm.hpp>

TEST(algorithm_host, canCompiletimeUnroll)
{
  int sum = 0;
  fluid::unrolled_for<3>([&sum] (auto i)
  {
    sum += i;
  });
  EXPECT_EQ(sum, 3);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
