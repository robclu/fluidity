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

#include <fluidity/container/host_tensor.hpp>
#include <fluidity/iterator/range.hpp>
#include <fluidity/iterator/multidim_iterator.hpp>
#include <gtest/gtest.h>

TEST(range_host_tests, can_create_simple_range)
{
  int i = 0;
  for (auto r : fluid::range(100))
  {
    EXPECT_EQ(r, i++);
  }
}

TEST(range_host_tests, can_create_stepped_range)
{
  int i = 10, end = 100, step = 2;
  for (auto r : fluid::range(i, end, step))
  {
    EXPECT_EQ(r, i);
    i += step;
  }
}

TEST(range_host_tests, range_works_with_non_integer_types)
{
  float i = 0.3f, end = 0.9f, step = 0.1f;
  for (auto f : fluid::range(i, end, step)) 
  {
    EXPECT_EQ(f, i);
    i += step;
  }
}

TEST(multidim_iter_host_tests, can_create_and_iterate_multidim_iterator)
{
  using namespace fluid;

  // Create information to define 3 x 2 x 2 dimensional space
  using dim_info_2dt = DimInfoCt<4, 3>;
  using dim_info_3dt = DimInfoCt<3, 2, 2>;  // 3x2x2 dimensional space

  using multi_iter_2dt = MultidimIterator<std::size_t, dim_info_2dt>;
  using multi_iter_3dt = MultidimIterator<std::size_t, dim_info_3dt>;

  EXPECT_EQ(dim_info_2dt::total_size(), dim_info_3dt::total_size());
  constexpr auto size = dim_info_2dt::total_size();

  std::size_t data[size];
  for (const auto i : range(size)) { data[i] = i; }

  // Create iterator over the space, the data will look like:
  //
  //       |- x x x -| |- x  x  x -|
  //  ---------------------------------
  //    y  |  0 1 2       6  7  8
  //    y  |  3 4 5       9 10 11
  //        |---z---|    |---z---|
  multi_iter_3dt iter3d(data);

  EXPECT_EQ(multi_iter_3dt::stride(fluid::dim_x), 1);
  EXPECT_EQ(multi_iter_3dt::stride(fluid::dim_y), 3);
  EXPECT_EQ(multi_iter_3dt::stride(fluid::dim_z), 6);

  EXPECT_EQ(*iter3d                        , 0);
  EXPECT_EQ(*iter3d.offset(1, fluid::dim_x), 1);
  EXPECT_EQ(*iter3d.offset(2, fluid::dim_x), 2);

  // Move to second y index:
  iter3d.shift(1, fluid::dim_y);
  EXPECT_EQ(*iter3d                        , 3);
  EXPECT_EQ(*iter3d.offset(1, fluid::dim_x), 4);
  EXPECT_EQ(*iter3d.offset(2, fluid::dim_x), 5);

  // Move to z1:
  iter3d.shift(1, fluid::dim_z).shift(-1, fluid::dim_y);
  EXPECT_EQ(*iter3d                        , 6);
  EXPECT_EQ(*iter3d.offset(1, fluid::dim_x), 7);
  EXPECT_EQ(*iter3d.offset(2, fluid::dim_x), 8);

  // Move to second y index:
  iter3d.shift(1, fluid::dim_y);
  EXPECT_EQ(*iter3d                        , 9);
  EXPECT_EQ(*iter3d.offset(1, fluid::dim_x), 10);
  EXPECT_EQ(*iter3d.offset(2, fluid::dim_x), 11);

  // Change to the 2D representation (same data):
  //
  //       |- x x  x x -|
  //  ------------------------
  //    y  |  0 1  2  3
  //    y  |  4 5  6  7
  //    y  |  8 9 10 11
  multi_iter_2dt iter2d(data);

  EXPECT_EQ(multi_iter_2dt::stride(fluid::dim_x), 1);
  EXPECT_EQ(multi_iter_2dt::stride(fluid::dim_y), 4);

  EXPECT_EQ(*iter2d                        , 0);
  EXPECT_EQ(*iter2d.offset(1, fluid::dim_x), 1);
  EXPECT_EQ(*iter2d.offset(2, fluid::dim_x), 2);
  EXPECT_EQ(*iter2d.offset(3, fluid::dim_x), 3);

  iter2d.shift(1, fluid::dim_y);
  EXPECT_EQ(*iter2d                        , 4);
  EXPECT_EQ(*iter2d.offset(1, fluid::dim_x), 5);
  EXPECT_EQ(*iter2d.offset(2, fluid::dim_x), 6);
  EXPECT_EQ(*iter2d.offset(3, fluid::dim_x), 7);

  iter2d.shift(1, fluid::dim_y);
  EXPECT_EQ(*iter2d                        , 8 );
  EXPECT_EQ(*iter2d.offset(1, fluid::dim_x), 9 );
  EXPECT_EQ(*iter2d.offset(2, fluid::dim_x), 10);
  EXPECT_EQ(*iter2d.offset(3, fluid::dim_x), 11);
}

/*
TEST(multidim_iter_host_tests, can_use_default_dimensions)
{
  using namespace fluid;

  using dim_info_1dt   = DimInfoCt<8>;
  using dim_info_2dt   = DimInfoCt<4, 2>;
  using multi_iter_1dt = MultidimIterator<std::size_t, dim_info_1dt>;

  EXPECT_EQ(dim_info_2dt::total_size(), dim_info_1dt::total_size());
  constexpr auto size = dim_info_1dt::total_size();

  std::size_t data[size];
  for (const auto i : range(size)) { data[i] = i; }

  // Create iterator over the space, the data will look like:
  //
  //       |- x x x x x x x x x x  x  x -|
  //  ------------------------------------
  //    y  |  0 1 2 3 4 5 6 7 8 9 10 11
  multi_iter_1dt iter1d(data);
  EXPECT_EQ(multi_iter_1dt::stride(fluid::dim_x), 1);

  EXPECT_EQ(*iter1d                        , 0);
  EXPECT_EQ(*iter1d.offset(1, fluid::dim_x), 1);
  EXPECT_EQ(*iter1d.offset(2, fluid::dim_x), 2);
  EXPECT_EQ(*iter1d.offset(3, fluid::dim_x), 3);
  EXPECT_EQ(*iter1d.offset(4, fluid::dim_x), 4);
  EXPECT_EQ(*iter1d.offset(5, fluid::dim_x), 5);
  EXPECT_EQ(*iter1d.offset(6, fluid::dim_x), 6);
  EXPECT_EQ(*iter1d.offset(7, fluid::dim_x), 7);

  EXPECT_EQ(*iter1d.offset(1), 1);
  EXPECT_EQ(*iter1d.offset(2), 2);
  EXPECT_EQ(*iter1d.offset(3), 3);
  EXPECT_EQ(*iter1d.offset(4), 4);
  EXPECT_EQ(*iter1d.offset(5), 5);
  EXPECT_EQ(*iter1d.offset(6), 6);
  EXPECT_EQ(*iter1d.offset(7), 7);
}
*/

TEST(multidim_iter_host_tests, can_compute_iterator_differences)
{
  using namespace fluid;

  // Create information to define 3 x 3 x 3 dimensional space
  using dim_info_t   = DimInfoCt<3, 3, 3>;  // 3x2x2 dimensional space
  using multi_iter_t = MultidimIterator<std::size_t, dim_info_t>;

  constexpr auto size = dim_info_t::total_size();

  std::size_t data[size];
  for (const auto i : range(size)) { data[i] = i; }

  // Create iterator over the space, the data will look like:
  //
  //       |- x x x -| |- x  x  x -| |-  x  x  x -| 
  //  -----------------------------------------------
  //    y  |  0 1 2       9  10 11      18 19 20
  //    y  |  3 4 5       12 13 14      21 22 23
  //    y  |  6 7 8       15 16 17      24 25 26
  //        |---z---|    |----z---|    |----z---|
  multi_iter_t iter(data);

  // Move the iterator to the center of the space.
  iter.shift(1, fluid::dim_x).shift(1, fluid::dim_y).shift(1, fluid::dim_z);

  EXPECT_EQ(iter.backward_diff(fluid::dim_x), 13 - 12);
  EXPECT_EQ(iter.forward_diff(fluid::dim_x) , 14 - 13); 
  EXPECT_EQ(iter.central_diff(fluid::dim_x) , 14 - 12);

  EXPECT_EQ(iter.backward_diff(fluid::dim_y), 13 - 10);
  EXPECT_EQ(iter.forward_diff(fluid::dim_y) , 16 - 13); 
  EXPECT_EQ(iter.central_diff(fluid::dim_y) , 16 - 10);

  EXPECT_EQ(iter.backward_diff(fluid::dim_z), 13 - 4 );
  EXPECT_EQ(iter.forward_diff(fluid::dim_z) , 22 - 13); 
  EXPECT_EQ(iter.central_diff(fluid::dim_z) , 22 - 4 ); 
}



int main(int argc, char** argv) 
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}