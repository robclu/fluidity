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

#include <fluidity/algorithm/fill.hpp>
#include <fluidity/container/host_tensor.hpp>
#include <fluidity/container/array.hpp>
#include <gtest/gtest.h>

template <typename T>
using host_tensor1d = fluid::HostTensor<T, 1>;
using array_t       = fluid::Array<int, 3>;

TEST(container_host_tensor, canCreateTensor)
{
  host_tensor1d<float> t(20);

  EXPECT_EQ(t.size(), static_cast<decltype(t.size())>(20));
}

TEST(container_host_tensor, canFillTensor)
{
  host_tensor1d<int> t(20);
  fluid::fill(t.begin(), t.end(), 2);

  for (const auto& element : t) {
    EXPECT_EQ(element, 2);
  }
}

TEST(container_host_tensor, canFillTensorWithFunctor)
{
  host_tensor1d<int> t(20);
  std::size_t count = 0;
  fluid::fill(t.begin(), t.end(), [&count] (auto iterator)
  {
    *iterator = count++;
  });

  count = 0;
  for (const auto& element : t) {
    EXPECT_EQ(element, count++);
  }
}

TEST(container_array, canCreateArray)
{
  array_t a{2};

  EXPECT_EQ(a.size(), 3);
  EXPECT_EQ(a[0], 2);
  EXPECT_EQ(a[1], 2);
  EXPECT_EQ(a[2], 2);
}

TEST(container_array, canCopyArray)
{
  array_t a{2};
  array_t b{a};
  auto c = b;

  EXPECT_EQ(b.size(), 3);
  EXPECT_EQ(b[0], 2);
  EXPECT_EQ(b[1], 2);
  EXPECT_EQ(b[2], 2);

  EXPECT_EQ(c.size(), 3);
  EXPECT_EQ(c[0], 2);
  EXPECT_EQ(c[1], 2);
  EXPECT_EQ(c[2], 2);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
