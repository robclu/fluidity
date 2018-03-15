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

TEST(algorithm_host_tests, can_compile_time_unroll)
{
  constexpr std::size_t amount = 3;
  int sum = 0;
  fluid::unrolled_for<amount>([&sum] (auto i)
  {
    sum += i;
  });
  EXPECT_EQ(sum, amount);
}

TEST(algorithm_host_tests, can_compile_time_unroll_above_max_unroll_depth)
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

TEST(algorithm_host_tests, can_use_if_constexpr)
{
  int x = 0;

  constexpr int true_value  = 10;
  constexpr int false_value = 3;
  fluid::if_constexpr<true>
  (
    [&]
    {
      x = true_value;
    },
    [&]
    {
      x = false_value;
    }
  );
  EXPECT_EQ(x, true_value);

  fluid::if_constexpr<false>
  (
    [&]
    {
      x = true_value;
    },
    [&]
    {
      x = false_value;
    }
  );
  EXPECT_EQ(x, false_value);
}

template <bool C>
struct TestWrapper {
  template <typename It>
  void apply(It iterator, int& value)
  {
    fluid::if_constexpr<C>
    (
      [&]
      {
        value = *iterator;
      },
      [&]
      {
        iterator = value * 2;
      }
    );
  }
};

TEST(algorithm_host_tests, if_constexpr_handles_different_semantics)
{
  struct Iterator {
    int* ptr;

    int& operator*() { return *ptr; }
    void operator=(int x) { *ptr = x; }
  };

  int x = 4;
  int y = 0;
  Iterator it{&x};

  TestWrapper<true>  true_test;
  TestWrapper<false> false_test;

  false_test.apply(it, y);
  EXPECT_EQ(x, y * 2);

  x = 18;
  true_test.apply(it, y);
  EXPECT_EQ(y, x);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
