//==--- fluidity/tests/iterator_tests_host.cpp ------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  iterator_tests_host.cpp
/// \brief This file defines tests for host side iterator functionality.
//
//==------------------------------------------------------------------------==//

#include <fluidity/iterator/range.hpp>
#include <gtest/gtest.h>

TEST(range_host_tests, canCreateSimpleRange) {
  int i = 0;
  for (auto r : fluid::range(100)) {
    EXPECT_EQ(r, i++);
  }
}

TEST(range_host_tests, canCreateSteppedRange) {
  int i = 10, end = 100, step = 2;
  for (auto r : fluid::range(i, end, step)) {
    EXPECT_EQ(r, i);
    i += step;
  }
}

TEST(range_host_tests, rangeWorksWithNonIntegerTypes) {
  float i = 0.3f, end = 0.9f, step = 0.1f;
  for (auto f : fluid::range(i, end, step)) {
    EXPECT_EQ(f, i);
    i += step;
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}