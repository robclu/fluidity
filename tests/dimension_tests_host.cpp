//==--- fluidity/tests/dimension_tests_host.cpp ------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  dimensions_tests_host.cpp
/// \brief This file defines tests for dimension related functionality.
//
//==------------------------------------------------------------------------==//

#include <fluidity/dimension/dimension.hpp>
#include <fluidity/dimension/dimension_info.hpp>
#include <gtest/gtest.h>

TEST(dimension_tests_host, dimenison_alias_values_are_correct)
{
  EXPECT_EQ(fluid::dim_x, 0);
  EXPECT_EQ(fluid::dim_y, 1);
  EXPECT_EQ(fluid::dim_z, 2);
}

TEST(dimension_info_tests_host, sizes_compute_correctly)
{
  constexpr auto size_x = 4, size_y = 5, size_z = 6;
  using dim_info_t = fluid::DimInfoCt<size_x, size_y, size_z>;

  EXPECT_EQ(dim_info_t::size(fluid::dim_x), size_x);
  EXPECT_EQ(dim_info_t::size(fluid::dim_y), size_y);
  EXPECT_EQ(dim_info_t::size(fluid::dim_z), size_z);
  EXPECT_EQ(dim_info_t::num_dimensions()  , 3     );
}

TEST(dimension_info_tests_host, total_size_computes_correctly)
{
  constexpr auto size_x = 7, size_y = 5, size_z = 6;
  using dim_info_t = fluid::DimInfoCt<size_x, size_y, size_z>;

  EXPECT_EQ(dim_info_t::total_size(), size_x * size_y * size_z);
}

int main(int argc, char** argv) 
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
