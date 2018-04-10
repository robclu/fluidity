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

  EXPECT_EQ(dim_info_t().size(fluid::dim_x), size_x);
  EXPECT_EQ(dim_info_t().size(fluid::dim_y), size_y);
  EXPECT_EQ(dim_info_t().size(fluid::dim_z), size_z);
  EXPECT_EQ(dim_info_t().num_dimensions()  , 3     );
}

TEST(dimension_info_tests_host, sizes_compute_correctly_for_runtime_impl)
{
  constexpr auto size_x = 4, size_y = 5, size_z = 6;
  using dim_info_t = fluid::DimInfo;

  dim_info_t dim_info(size_x, size_y, size_z);

  EXPECT_EQ(dim_info.size(fluid::dim_x), size_x);
  EXPECT_EQ(dim_info.size(fluid::dim_y), size_y);
  EXPECT_EQ(dim_info.size(fluid::dim_z), size_z);
  EXPECT_EQ(dim_info.num_dimensions()  , 3     );

  dim_info.push_back(4);
  EXPECT_EQ(dim_info.size(fluid::Dimension<3>{}), 4);
  EXPECT_EQ(dim_info.num_dimensions()           , 4);
}

TEST(dimension_info_tests_host, total_size_computes_correctly_ct)
{
  constexpr auto size_x = 7, size_y = 5, size_z = 6;
  using dim_info_t = fluid::DimInfoCt<size_x, size_y, size_z>;

  EXPECT_EQ(dim_info_t().total_size(), size_x * size_y * size_z);
}

TEST(dimension_info_tests_host, total_size_computes_correctly_rt)
{
  constexpr auto size_x = 7, size_y = 5, size_z = 6;
  using dim_info_t = fluid::DimInfo;

  dim_info_t dim_info(size_x, size_y, size_z);

  EXPECT_EQ(dim_info.total_size(), size_x * size_y * size_z);
}

TEST(dimension_info_tests_host, can_get_flattened_indices_ct)
{
  constexpr auto size_x = 3, size_y = 3, size_z = 2;
  using dim_info_t = fluid::DimInfoCt<size_x, size_y, size_z>;

  EXPECT_EQ(dim_info_t().flattened_index(0 , fluid::dim_x), 0);
  EXPECT_EQ(dim_info_t().flattened_index(1 , fluid::dim_x), 1);
  EXPECT_EQ(dim_info_t().flattened_index(2 , fluid::dim_x), 2);
  EXPECT_EQ(dim_info_t().flattened_index(3 , fluid::dim_x), 0);
  EXPECT_EQ(dim_info_t().flattened_index(4 , fluid::dim_x), 1);
  EXPECT_EQ(dim_info_t().flattened_index(5 , fluid::dim_x), 2);
  EXPECT_EQ(dim_info_t().flattened_index(6 , fluid::dim_x), 0);
  EXPECT_EQ(dim_info_t().flattened_index(7 , fluid::dim_x), 1);
  EXPECT_EQ(dim_info_t().flattened_index(8 , fluid::dim_x), 2);
  EXPECT_EQ(dim_info_t().flattened_index(9 , fluid::dim_x), 0);
  EXPECT_EQ(dim_info_t().flattened_index(10, fluid::dim_x), 1);
  EXPECT_EQ(dim_info_t().flattened_index(11, fluid::dim_x), 2);
  EXPECT_EQ(dim_info_t().flattened_index(12, fluid::dim_x), 0);
  EXPECT_EQ(dim_info_t().flattened_index(13, fluid::dim_x), 1);
  EXPECT_EQ(dim_info_t().flattened_index(14, fluid::dim_x), 2);
  EXPECT_EQ(dim_info_t().flattened_index(15, fluid::dim_x), 0);
  EXPECT_EQ(dim_info_t().flattened_index(16, fluid::dim_x), 1);
  EXPECT_EQ(dim_info_t().flattened_index(17, fluid::dim_x), 2);

  EXPECT_EQ(dim_info_t().flattened_index(0 , fluid::dim_y), 0);
  EXPECT_EQ(dim_info_t().flattened_index(1 , fluid::dim_y), 0);
  EXPECT_EQ(dim_info_t().flattened_index(2 , fluid::dim_y), 0);
  EXPECT_EQ(dim_info_t().flattened_index(3 , fluid::dim_y), 1);
  EXPECT_EQ(dim_info_t().flattened_index(4 , fluid::dim_y), 1);
  EXPECT_EQ(dim_info_t().flattened_index(5 , fluid::dim_y), 1);
  EXPECT_EQ(dim_info_t().flattened_index(6 , fluid::dim_y), 2);
  EXPECT_EQ(dim_info_t().flattened_index(7 , fluid::dim_y), 2);
  EXPECT_EQ(dim_info_t().flattened_index(8 , fluid::dim_y), 2);
  EXPECT_EQ(dim_info_t().flattened_index(9 , fluid::dim_y), 0);
  EXPECT_EQ(dim_info_t().flattened_index(10, fluid::dim_y), 0);
  EXPECT_EQ(dim_info_t().flattened_index(11, fluid::dim_y), 0);
  EXPECT_EQ(dim_info_t().flattened_index(12, fluid::dim_y), 1);
  EXPECT_EQ(dim_info_t().flattened_index(13, fluid::dim_y), 1);
  EXPECT_EQ(dim_info_t().flattened_index(14, fluid::dim_y), 1);
  EXPECT_EQ(dim_info_t().flattened_index(15, fluid::dim_y), 2);
  EXPECT_EQ(dim_info_t().flattened_index(16, fluid::dim_y), 2);
  EXPECT_EQ(dim_info_t().flattened_index(17, fluid::dim_y), 2);

  EXPECT_EQ(dim_info_t().flattened_index(0 , fluid::dim_z), 0);
  EXPECT_EQ(dim_info_t().flattened_index(1 , fluid::dim_z), 0);
  EXPECT_EQ(dim_info_t().flattened_index(2 , fluid::dim_z), 0);
  EXPECT_EQ(dim_info_t().flattened_index(3 , fluid::dim_z), 0);
  EXPECT_EQ(dim_info_t().flattened_index(4 , fluid::dim_z), 0);
  EXPECT_EQ(dim_info_t().flattened_index(5 , fluid::dim_z), 0);
  EXPECT_EQ(dim_info_t().flattened_index(6 , fluid::dim_z), 0);
  EXPECT_EQ(dim_info_t().flattened_index(7 , fluid::dim_z), 0);
  EXPECT_EQ(dim_info_t().flattened_index(8 , fluid::dim_z), 0);
  EXPECT_EQ(dim_info_t().flattened_index(9 , fluid::dim_z), 1);
  EXPECT_EQ(dim_info_t().flattened_index(10, fluid::dim_z), 1);
  EXPECT_EQ(dim_info_t().flattened_index(11, fluid::dim_z), 1);
  EXPECT_EQ(dim_info_t().flattened_index(12, fluid::dim_z), 1);
  EXPECT_EQ(dim_info_t().flattened_index(13, fluid::dim_z), 1);
  EXPECT_EQ(dim_info_t().flattened_index(14, fluid::dim_z), 1);
  EXPECT_EQ(dim_info_t().flattened_index(15, fluid::dim_z), 1);
  EXPECT_EQ(dim_info_t().flattened_index(16, fluid::dim_z), 1);
  EXPECT_EQ(dim_info_t().flattened_index(17, fluid::dim_z), 1);
}

TEST(dimension_info_tests_host, can_get_flattened_indices_rt)
{
  constexpr auto size_x = 3, size_y = 3, size_z = 2;
  using dim_info_t = fluid::DimInfo;

  dim_info_t dim_info(size_x, size_y, size_z);

  EXPECT_EQ(dim_info.flattened_index(0 , fluid::dim_x), 0);
  EXPECT_EQ(dim_info.flattened_index(1 , fluid::dim_x), 1);
  EXPECT_EQ(dim_info.flattened_index(2 , fluid::dim_x), 2);
  EXPECT_EQ(dim_info.flattened_index(3 , fluid::dim_x), 0);
  EXPECT_EQ(dim_info.flattened_index(4 , fluid::dim_x), 1);
  EXPECT_EQ(dim_info.flattened_index(5 , fluid::dim_x), 2);
  EXPECT_EQ(dim_info.flattened_index(6 , fluid::dim_x), 0);
  EXPECT_EQ(dim_info.flattened_index(7 , fluid::dim_x), 1);
  EXPECT_EQ(dim_info.flattened_index(8 , fluid::dim_x), 2);
  EXPECT_EQ(dim_info.flattened_index(9 , fluid::dim_x), 0);
  EXPECT_EQ(dim_info.flattened_index(10, fluid::dim_x), 1);
  EXPECT_EQ(dim_info.flattened_index(11, fluid::dim_x), 2);
  EXPECT_EQ(dim_info.flattened_index(12, fluid::dim_x), 0);
  EXPECT_EQ(dim_info.flattened_index(13, fluid::dim_x), 1);
  EXPECT_EQ(dim_info.flattened_index(14, fluid::dim_x), 2);
  EXPECT_EQ(dim_info.flattened_index(15, fluid::dim_x), 0);
  EXPECT_EQ(dim_info.flattened_index(16, fluid::dim_x), 1);
  EXPECT_EQ(dim_info.flattened_index(17, fluid::dim_x), 2);

  EXPECT_EQ(dim_info.flattened_index(0 , fluid::dim_y), 0);
  EXPECT_EQ(dim_info.flattened_index(1 , fluid::dim_y), 0);
  EXPECT_EQ(dim_info.flattened_index(2 , fluid::dim_y), 0);
  EXPECT_EQ(dim_info.flattened_index(3 , fluid::dim_y), 1);
  EXPECT_EQ(dim_info.flattened_index(4 , fluid::dim_y), 1);
  EXPECT_EQ(dim_info.flattened_index(5 , fluid::dim_y), 1);
  EXPECT_EQ(dim_info.flattened_index(6 , fluid::dim_y), 2);
  EXPECT_EQ(dim_info.flattened_index(7 , fluid::dim_y), 2);
  EXPECT_EQ(dim_info.flattened_index(8 , fluid::dim_y), 2);
  EXPECT_EQ(dim_info.flattened_index(9 , fluid::dim_y), 0);
  EXPECT_EQ(dim_info.flattened_index(10, fluid::dim_y), 0);
  EXPECT_EQ(dim_info.flattened_index(11, fluid::dim_y), 0);
  EXPECT_EQ(dim_info.flattened_index(12, fluid::dim_y), 1);
  EXPECT_EQ(dim_info.flattened_index(13, fluid::dim_y), 1);
  EXPECT_EQ(dim_info.flattened_index(14, fluid::dim_y), 1);
  EXPECT_EQ(dim_info.flattened_index(15, fluid::dim_y), 2);
  EXPECT_EQ(dim_info.flattened_index(16, fluid::dim_y), 2);
  EXPECT_EQ(dim_info.flattened_index(17, fluid::dim_y), 2);

  EXPECT_EQ(dim_info.flattened_index(0 , fluid::dim_z), 0);
  EXPECT_EQ(dim_info.flattened_index(1 , fluid::dim_z), 0);
  EXPECT_EQ(dim_info.flattened_index(2 , fluid::dim_z), 0);
  EXPECT_EQ(dim_info.flattened_index(3 , fluid::dim_z), 0);
  EXPECT_EQ(dim_info.flattened_index(4 , fluid::dim_z), 0);
  EXPECT_EQ(dim_info.flattened_index(5 , fluid::dim_z), 0);
  EXPECT_EQ(dim_info.flattened_index(6 , fluid::dim_z), 0);
  EXPECT_EQ(dim_info.flattened_index(7 , fluid::dim_z), 0);
  EXPECT_EQ(dim_info.flattened_index(8 , fluid::dim_z), 0);
  EXPECT_EQ(dim_info.flattened_index(9 , fluid::dim_z), 1);
  EXPECT_EQ(dim_info.flattened_index(10, fluid::dim_z), 1);
  EXPECT_EQ(dim_info.flattened_index(11, fluid::dim_z), 1);
  EXPECT_EQ(dim_info.flattened_index(12, fluid::dim_z), 1);
  EXPECT_EQ(dim_info.flattened_index(13, fluid::dim_z), 1);
  EXPECT_EQ(dim_info.flattened_index(14, fluid::dim_z), 1);
  EXPECT_EQ(dim_info.flattened_index(15, fluid::dim_z), 1);
  EXPECT_EQ(dim_info.flattened_index(16, fluid::dim_z), 1);
  EXPECT_EQ(dim_info.flattened_index(17, fluid::dim_z), 1);
}

int main(int argc, char** argv) 
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
