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

TEST(algorithm_host, can_compile_time_unroll)
{
  constexpr std::size_t amount = 3;
  int sum = 0;
  fluid::unrolled_for<amount>([&sum] (auto i)
  {
    sum += i;
  });
  EXPECT_EQ(sum, amount);
}

TEST(algorithm_host, can_compile_time_unroll_above_max_unroll_depth)
{
  constexpr std::size_t amount = 64;
  static_assert(amount > fluid::max_unroll_depth,
                "Test is useless, MAX_UNROLL_DEPTH is extremely high!");

  int sum = 0, result = 0;
  fluid::unrolled_for_bounded<amount>([&sum] (auto i)
  {
    sum += i;
  });

  for (const auto i : fluid::range(amount))
  {
    result += i;
  }
  EXPECT_EQ(sum, result);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
