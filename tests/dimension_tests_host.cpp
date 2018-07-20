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
  EXPECT_EQ(fluid::dim_x, std::size_t{0});
  EXPECT_EQ(fluid::dim_y, std::size_t{1});
  EXPECT_EQ(fluid::dim_z, std::size_t{2});
}

TEST(dimension_info_tests_host, sizes_compute_correctly)
{
  constexpr std::size_t size_x = 4, size_y = 5, size_z = 6;
  using dim_info_t = fluid::DimInfoCt<size_x, size_y, size_z>;

  EXPECT_EQ(dim_info_t().size(fluid::dim_x), size_x);
  EXPECT_EQ(dim_info_t().size(fluid::dim_y), size_y);
  EXPECT_EQ(dim_info_t().size(fluid::dim_z), size_z);
  EXPECT_EQ(dim_info_t().num_dimensions()  , std::size_t{3});
}

TEST(dimension_info_tests_host, sizes_compute_correctly_for_runtime_impl)
{
  constexpr std::size_t size_x = 4, size_y = 5, size_z = 6, dims = 3;
  using dim_info_t = fluid::DimInfo<dims>;

  dim_info_t dim_info(size_x, size_y, size_z);

  EXPECT_EQ(dim_info.size(fluid::dim_x), size_x);
  EXPECT_EQ(dim_info.size(fluid::dim_y), size_y);
  EXPECT_EQ(dim_info.size(fluid::dim_z), size_z);
  EXPECT_EQ(dim_info.num_dimensions()  , dims  );

  EXPECT_EQ(dim_info.size(fluid::Dimension<0>{}), size_x);
  EXPECT_EQ(dim_info.size(fluid::Dimension<1>{}), size_y);
  EXPECT_EQ(dim_info.size(fluid::Dimension<2>{}), size_z);

  dim_info[0] = size_x * size_x;
  dim_info[1] = size_y * size_y;
  dim_info[2] = size_z * size_z;

  EXPECT_EQ(dim_info.size(fluid::Dimension<0>{}), size_x * size_x);
  EXPECT_EQ(dim_info.size(fluid::Dimension<1>{}), size_y * size_y);
  EXPECT_EQ(dim_info.size(fluid::Dimension<2>{}), size_z * size_z);
}

TEST(dimension_info_tests_host, total_size_ct)
{
  constexpr std::size_t size_x = 7, size_y = 5, size_z = 6;
  using dim_info_t = fluid::DimInfoCt<size_x, size_y, size_z>;

  EXPECT_EQ(dim_info_t().total_size(), size_x * size_y * size_z);
}

TEST(dimension_info_tests_host, total_size_uniform_padding_ct)
{
  constexpr std::size_t size_x = 7, size_y = 5, size_z = 6, padding = 2;
  constexpr std::size_t ptotal = padding << 1;
  using dim_info_t = fluid::DimInfoCt<size_x, size_y, size_z>;

  EXPECT_EQ(dim_info_t().total_size<padding>(), 
            (size_x + ptotal) * (size_y + ptotal) * (size_z + ptotal));
}

TEST(dimension_info_tests_host, total_size_single_dim_padding_ct)
{
  using padding_xt = fluid::PaddingInfo<0, 2>;
  using padding_yt = fluid::PaddingInfo<1, 3>;
  using padding_zt = fluid::PaddingInfo<2, 1>;
  constexpr std::size_t size_x = 7, size_y = 5, size_z = 6;

  using dim_info_t = fluid::DimInfoCt<size_x, size_y, size_z>;

  EXPECT_EQ(dim_info_t().total_size<padding_xt>(), 
            (size_x + padding_xt::amount * 2) * size_y * size_z);
  EXPECT_EQ(dim_info_t().total_size<padding_yt>(), 
            size_x * (size_y + padding_yt::amount * 2) * size_z);
  EXPECT_EQ(dim_info_t().total_size<padding_zt>(), 
            size_x * size_y * (size_z + padding_zt::amount * 2));
}

TEST(dimension_info_tests_host, total_size_rt)
{
  constexpr std::size_t size_x = 7, size_y = 5, size_z = 6, dims = 3;
  using dim_info_t = fluid::DimInfo<dims>;

  dim_info_t dim_info(size_x, size_y, size_z);
  EXPECT_EQ(dim_info.total_size(), size_x * size_y * size_z);
}

TEST(dimension_info_tests_host, total_size_uniform_padding_rt)
{
  constexpr std::size_t size_x  = 7,
                        size_y  = 5,
                        size_z  = 6,
                        dims    = 3,
                        padding = 2;
  constexpr std::size_t ptotal  = padding << 1;
  using dim_info_t = fluid::DimInfo<dims>;

  dim_info_t dim_info(size_x, size_y, size_z);
  EXPECT_EQ(dim_info.total_size<padding>(), 
            (size_x + ptotal) * (size_y + ptotal) * (size_z + ptotal));
}

TEST(dimension_info_tests_host, total_size_single_dim_padding_rt)
{
  using padding_xt = fluid::PaddingInfo<0, 2>;
  using padding_yt = fluid::PaddingInfo<1, 3>;
  using padding_zt = fluid::PaddingInfo<2, 1>;
  constexpr std::size_t size_x = 7, size_y = 5, size_z = 6, dims = 3;

  using dim_info_t = fluid::DimInfo<dims>;
  dim_info_t dim_info(size_x, size_y, size_z);

  EXPECT_EQ(dim_info.total_size<padding_xt>(), 
            (size_x + padding_xt::amount * 2) * size_y * size_z);
  EXPECT_EQ(dim_info.total_size<padding_yt>(), 
            size_x * (size_y + padding_yt::amount * 2) * size_z);
  EXPECT_EQ(dim_info.total_size<padding_zt>(), 
            size_x * size_y * (size_z + padding_zt::amount * 2));
}

TEST(dimension_info_tests_host, can_get_flattened_indices_ct)
{
  constexpr std::size_t size_x = 3, size_y = 3, size_z = 2;
  using dim_info_t = fluid::DimInfoCt<size_x, size_y, size_z>;

  EXPECT_EQ(dim_info_t().flattened_index(0 , fluid::dim_x), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(1 , fluid::dim_x), std::size_t{1});
  EXPECT_EQ(dim_info_t().flattened_index(2 , fluid::dim_x), std::size_t{2});
  EXPECT_EQ(dim_info_t().flattened_index(3 , fluid::dim_x), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(4 , fluid::dim_x), std::size_t{1});
  EXPECT_EQ(dim_info_t().flattened_index(5 , fluid::dim_x), std::size_t{2});
  EXPECT_EQ(dim_info_t().flattened_index(6 , fluid::dim_x), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(7 , fluid::dim_x), std::size_t{1});
  EXPECT_EQ(dim_info_t().flattened_index(8 , fluid::dim_x), std::size_t{2});
  EXPECT_EQ(dim_info_t().flattened_index(9 , fluid::dim_x), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(10, fluid::dim_x), std::size_t{1});
  EXPECT_EQ(dim_info_t().flattened_index(11, fluid::dim_x), std::size_t{2});
  EXPECT_EQ(dim_info_t().flattened_index(12, fluid::dim_x), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(13, fluid::dim_x), std::size_t{1});
  EXPECT_EQ(dim_info_t().flattened_index(14, fluid::dim_x), std::size_t{2});
  EXPECT_EQ(dim_info_t().flattened_index(15, fluid::dim_x), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(16, fluid::dim_x), std::size_t{1});
  EXPECT_EQ(dim_info_t().flattened_index(17, fluid::dim_x), std::size_t{2});

  EXPECT_EQ(dim_info_t().flattened_index(0 , fluid::dim_y), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(1 , fluid::dim_y), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(2 , fluid::dim_y), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(3 , fluid::dim_y), std::size_t{1});
  EXPECT_EQ(dim_info_t().flattened_index(4 , fluid::dim_y), std::size_t{1});
  EXPECT_EQ(dim_info_t().flattened_index(5 , fluid::dim_y), std::size_t{1});
  EXPECT_EQ(dim_info_t().flattened_index(6 , fluid::dim_y), std::size_t{2});
  EXPECT_EQ(dim_info_t().flattened_index(7 , fluid::dim_y), std::size_t{2});
  EXPECT_EQ(dim_info_t().flattened_index(8 , fluid::dim_y), std::size_t{2});
  EXPECT_EQ(dim_info_t().flattened_index(9 , fluid::dim_y), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(10, fluid::dim_y), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(11, fluid::dim_y), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(12, fluid::dim_y), std::size_t{1});
  EXPECT_EQ(dim_info_t().flattened_index(13, fluid::dim_y), std::size_t{1});
  EXPECT_EQ(dim_info_t().flattened_index(14, fluid::dim_y), std::size_t{1});
  EXPECT_EQ(dim_info_t().flattened_index(15, fluid::dim_y), std::size_t{2});
  EXPECT_EQ(dim_info_t().flattened_index(16, fluid::dim_y), std::size_t{2});
  EXPECT_EQ(dim_info_t().flattened_index(17, fluid::dim_y), std::size_t{2});

  EXPECT_EQ(dim_info_t().flattened_index(0 , fluid::dim_z), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(1 , fluid::dim_z), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(2 , fluid::dim_z), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(3 , fluid::dim_z), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(4 , fluid::dim_z), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(5 , fluid::dim_z), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(6 , fluid::dim_z), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(7 , fluid::dim_z), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(8 , fluid::dim_z), std::size_t{0});
  EXPECT_EQ(dim_info_t().flattened_index(9 , fluid::dim_z), std::size_t{1});
  EXPECT_EQ(dim_info_t().flattened_index(10, fluid::dim_z), std::size_t{1});
  EXPECT_EQ(dim_info_t().flattened_index(11, fluid::dim_z), std::size_t{1});
  EXPECT_EQ(dim_info_t().flattened_index(12, fluid::dim_z), std::size_t{1});
  EXPECT_EQ(dim_info_t().flattened_index(13, fluid::dim_z), std::size_t{1});
  EXPECT_EQ(dim_info_t().flattened_index(14, fluid::dim_z), std::size_t{1});
  EXPECT_EQ(dim_info_t().flattened_index(15, fluid::dim_z), std::size_t{1});
  EXPECT_EQ(dim_info_t().flattened_index(16, fluid::dim_z), std::size_t{1});
  EXPECT_EQ(dim_info_t().flattened_index(17, fluid::dim_z), std::size_t{1});
}

TEST(dimension_info_tests_host, can_get_flattened_indices_rt)
{
  constexpr auto size_x = 3, size_y = 3, size_z = 2, dims = 3;
  using dim_info_t = fluid::DimInfo<dims>;

  dim_info_t dim_info(size_x, size_y, size_z);

  EXPECT_EQ(dim_info.flattened_index(0 , fluid::dim_x), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(1 , fluid::dim_x), std::size_t{1});
  EXPECT_EQ(dim_info.flattened_index(2 , fluid::dim_x), std::size_t{2});
  EXPECT_EQ(dim_info.flattened_index(3 , fluid::dim_x), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(4 , fluid::dim_x), std::size_t{1});
  EXPECT_EQ(dim_info.flattened_index(5 , fluid::dim_x), std::size_t{2});
  EXPECT_EQ(dim_info.flattened_index(6 , fluid::dim_x), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(7 , fluid::dim_x), std::size_t{1});
  EXPECT_EQ(dim_info.flattened_index(8 , fluid::dim_x), std::size_t{2});
  EXPECT_EQ(dim_info.flattened_index(9 , fluid::dim_x), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(10, fluid::dim_x), std::size_t{1});
  EXPECT_EQ(dim_info.flattened_index(11, fluid::dim_x), std::size_t{2});
  EXPECT_EQ(dim_info.flattened_index(12, fluid::dim_x), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(13, fluid::dim_x), std::size_t{1});
  EXPECT_EQ(dim_info.flattened_index(14, fluid::dim_x), std::size_t{2});
  EXPECT_EQ(dim_info.flattened_index(15, fluid::dim_x), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(16, fluid::dim_x), std::size_t{1});
  EXPECT_EQ(dim_info.flattened_index(17, fluid::dim_x), std::size_t{2});

  EXPECT_EQ(dim_info.flattened_index(0 , fluid::dim_y), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(1 , fluid::dim_y), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(2 , fluid::dim_y), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(3 , fluid::dim_y), std::size_t{1});
  EXPECT_EQ(dim_info.flattened_index(4 , fluid::dim_y), std::size_t{1});
  EXPECT_EQ(dim_info.flattened_index(5 , fluid::dim_y), std::size_t{1});
  EXPECT_EQ(dim_info.flattened_index(6 , fluid::dim_y), std::size_t{2});
  EXPECT_EQ(dim_info.flattened_index(7 , fluid::dim_y), std::size_t{2});
  EXPECT_EQ(dim_info.flattened_index(8 , fluid::dim_y), std::size_t{2});
  EXPECT_EQ(dim_info.flattened_index(9 , fluid::dim_y), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(10, fluid::dim_y), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(11, fluid::dim_y), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(12, fluid::dim_y), std::size_t{1});
  EXPECT_EQ(dim_info.flattened_index(13, fluid::dim_y), std::size_t{1});
  EXPECT_EQ(dim_info.flattened_index(14, fluid::dim_y), std::size_t{1});
  EXPECT_EQ(dim_info.flattened_index(15, fluid::dim_y), std::size_t{2});
  EXPECT_EQ(dim_info.flattened_index(16, fluid::dim_y), std::size_t{2});
  EXPECT_EQ(dim_info.flattened_index(17, fluid::dim_y), std::size_t{2});

  EXPECT_EQ(dim_info.flattened_index(0 , fluid::dim_z), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(1 , fluid::dim_z), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(2 , fluid::dim_z), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(3 , fluid::dim_z), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(4 , fluid::dim_z), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(5 , fluid::dim_z), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(6 , fluid::dim_z), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(7 , fluid::dim_z), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(8 , fluid::dim_z), std::size_t{0});
  EXPECT_EQ(dim_info.flattened_index(9 , fluid::dim_z), std::size_t{1});
  EXPECT_EQ(dim_info.flattened_index(10, fluid::dim_z), std::size_t{1});
  EXPECT_EQ(dim_info.flattened_index(11, fluid::dim_z), std::size_t{1});
  EXPECT_EQ(dim_info.flattened_index(12, fluid::dim_z), std::size_t{1});
  EXPECT_EQ(dim_info.flattened_index(13, fluid::dim_z), std::size_t{1});
  EXPECT_EQ(dim_info.flattened_index(14, fluid::dim_z), std::size_t{1});
  EXPECT_EQ(dim_info.flattened_index(15, fluid::dim_z), std::size_t{1});
  EXPECT_EQ(dim_info.flattened_index(16, fluid::dim_z), std::size_t{1});
  EXPECT_EQ(dim_info.flattened_index(17, fluid::dim_z), std::size_t{1});
}

int main(int argc, char** argv) 
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
