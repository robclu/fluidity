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

TEST(multidim_iter_host_tests, can_get_iterator_dimension_sizes)
{
  using namespace fluid;

  constexpr auto size_x = std::size_t{4};
  constexpr auto size_y = std::size_t{3};

  // Create information to define 3 x 2 x 2 dimensional space
  using dim_info_2d_ct_t = DimInfoCt<size_x, size_y>;
  using dim_info_2d_rt_t = DimInfo<2>;

  using multi_iter_2d_ct_t = MultidimIterator<std::size_t, dim_info_2d_ct_t>;
  using multi_iter_2d_rt_t = MultidimIterator<std::size_t, dim_info_2d_rt_t>;

  constexpr auto size = dim_info_2d_ct_t().total_size();
  std::size_t data[size];

  multi_iter_2d_ct_t iter2d_ct(data);
  EXPECT_EQ(iter2d_ct.size(dim_x), size_x);
  EXPECT_EQ(iter2d_ct.size(dim_y), size_y);

  multi_iter_2d_rt_t iter2d_rt(data, size_x, size_y);
  EXPECT_EQ(iter2d_rt.size(dim_x), size_x);
  EXPECT_EQ(iter2d_rt.size(dim_y), size_y);
}

TEST(multidim_iter_host_tests, can_create_and_iterate_multidim_iterator)
{
  using namespace fluid;

  constexpr auto size_x = std::size_t{4};
  constexpr auto size_y = std::size_t{3};

  // Create information to define 3 x 2 x 2 dimensional space
  using dim_info_2d_ct_t = DimInfoCt<size_x, size_y>;
  using dim_info_2d_rt_t = DimInfo<2>;

  using multi_iter_2d_ct_t = MultidimIterator<std::size_t, dim_info_2d_ct_t>;
  using multi_iter_2d_rt_t = MultidimIterator<std::size_t, dim_info_2d_rt_t>;

  constexpr auto size = dim_info_2d_ct_t().total_size();
  std::size_t data[size];
  for (const auto i : range(size)) { data[i] = i; }

  // Change to the 2D representation (same data):
  //
  //       |- x x  x x -|
  //  ------------------------
  //    y  |  0 1  2  3
  //    y  |  4 5  6  7
  //    y  |  8 9 10 11
  
  // Compile time version:
  multi_iter_2d_ct_t iter2d_ct(data);
  EXPECT_EQ(iter2d_ct.stride(fluid::dim_x), 1);
  EXPECT_EQ(iter2d_ct.stride(fluid::dim_y), size_x);

  EXPECT_EQ(*iter2d_ct                        , 0);
  EXPECT_EQ(*iter2d_ct.offset(1, fluid::dim_x), 1);
  EXPECT_EQ(*iter2d_ct.offset(2, fluid::dim_x), 2);
  EXPECT_EQ(*iter2d_ct.offset(3, fluid::dim_x), 3);

  iter2d_ct.shift(1, fluid::dim_y);
  EXPECT_EQ(*iter2d_ct                        , 4);
  EXPECT_EQ(*iter2d_ct.offset(1, fluid::dim_x), 5);
  EXPECT_EQ(*iter2d_ct.offset(2, fluid::dim_x), 6);
  EXPECT_EQ(*iter2d_ct.offset(3, fluid::dim_x), 7);

  iter2d_ct.shift(1, fluid::dim_y);
  EXPECT_EQ(*iter2d_ct                        , 8 );
  EXPECT_EQ(*iter2d_ct.offset(1, fluid::dim_x), 9 );
  EXPECT_EQ(*iter2d_ct.offset(2, fluid::dim_x), 10);
  EXPECT_EQ(*iter2d_ct.offset(3, fluid::dim_x), 11);

  // Runtime version:
  multi_iter_2d_rt_t iter2d_rt(data, size_x, size_y);
  EXPECT_EQ(iter2d_rt.stride(fluid::dim_x), 1);
  EXPECT_EQ(iter2d_rt.stride(fluid::dim_y), 4);

  EXPECT_EQ(*iter2d_rt                        , 0);
  EXPECT_EQ(*iter2d_rt.offset(1, fluid::dim_x), 1);
  EXPECT_EQ(*iter2d_rt.offset(2, fluid::dim_x), 2);
  EXPECT_EQ(*iter2d_rt.offset(3, fluid::dim_x), 3);

  iter2d_rt.shift(1, fluid::dim_y);
  EXPECT_EQ(*iter2d_rt                        , 4);
  EXPECT_EQ(*iter2d_rt.offset(1, fluid::dim_x), 5);
  EXPECT_EQ(*iter2d_rt.offset(2, fluid::dim_x), 6);
  EXPECT_EQ(*iter2d_rt.offset(3, fluid::dim_x), 7);

  iter2d_rt.shift(1, fluid::dim_y);
  EXPECT_EQ(*iter2d_rt                        , 8 );
  EXPECT_EQ(*iter2d_rt.offset(1, fluid::dim_x), 9 );
  EXPECT_EQ(*iter2d_rt.offset(2, fluid::dim_x), 10);
  EXPECT_EQ(*iter2d_rt.offset(3, fluid::dim_x), 11);
}

TEST(multidim_iter_host_tests, modified_iter_strides_are_the_same_ct)
{
  using namespace fluid;

  // Create information to define 3 x 3 x 3 dimensional space
  using dim_info_t   = DimInfoCt<3, 3, 3>;  // 3x2x2 dimensional space
  using multi_iter_t = MultidimIterator<std::size_t, dim_info_t>;

  constexpr auto size = dim_info_t().total_size();
  std::size_t data[size];
  for (const auto i : range(size)) { data[i] = i; }

  auto it_1 = multi_iter_t(data);
  auto it_2 = it_1.offset(1, dim_x);
  auto it_3 = it_2.offset(1, dim_y);
  auto it_4 = it_3.offset(1, dim_z);

  EXPECT_EQ(it_1.stride(dim_x), it_2.stride(dim_x));
  EXPECT_EQ(it_2.stride(dim_y), it_3.stride(dim_y));
  EXPECT_EQ(it_3.stride(dim_z), it_4.stride(dim_z));
}

TEST(multidim_iter_host_tests, modified_iter_strides_are_the_same_rt)
{
  using namespace fluid;

  // Create information to define 3 x 3 x 3 dimensional space
  using multi_iter_t = MultidimIterator<std::size_t, DimInfo<3>>;

  constexpr auto dim_size = std::size_t{3};
  constexpr auto size     = dim_size * dim_size * dim_size;
  std::size_t data[size];
  for (const auto i : range(size)) { data[i] = i; }

  auto it_1 = multi_iter_t(data, dim_size, dim_size, dim_size);
  auto it_2 = it_1.offset(1, dim_x);
  auto it_3 = it_2.offset(1, dim_y);
  auto it_4 = it_3.offset(1, dim_z);

  EXPECT_EQ(it_1.stride(dim_x), it_2.stride(dim_x));
  EXPECT_EQ(it_2.stride(dim_y), it_3.stride(dim_y));
  EXPECT_EQ(it_3.stride(dim_z), it_4.stride(dim_z));
}

TEST(multidim_iter_host_tests, can_compute_iterator_differences_ct_iterator)
{
  using namespace fluid;

  // Create information to define 3 x 3 x 3 dimensional space
  using dim_info_t   = DimInfoCt<3, 3, 3>;  // 3x2x2 dimensional space
  using multi_iter_t = MultidimIterator<std::size_t, dim_info_t>;

  constexpr auto size = dim_info_t().total_size();
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

TEST(multidim_iter_host_tests, can_compute_iterator_differences_rt_iterator)
{
  using namespace fluid;
  using multi_iter_t = MultidimIterator<std::size_t, DimInfo<3>>;
  
  constexpr auto dim_size = std::size_t{3};
  constexpr auto size     = dim_size * dim_size * dim_size;
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
  multi_iter_t iter(data, dim_size, dim_size, dim_size);

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

TEST(strided_iter_host_tests, iterates_correctly_for_linear_data)
{
  using namespace fluid;

  using dim_info_1dt   = DimInfoCt<8>;
  using dim_info_2dt   = DimInfoCt<4, 2>;
  using multi_iter_1dt = MultidimIterator<std::size_t, dim_info_1dt>;

  EXPECT_EQ(dim_info_2dt().total_size(), dim_info_1dt().total_size());

  constexpr auto size = dim_info_1dt().total_size();
  std::size_t data[size];
  for (const auto i : range(size)) { data[i] = i; }

  // Create iterator over the space, the data will look like:
  //
  //       |- x x x x x x x x x x  x  x -|
  //  ------------------------------------
  //    y  |  0 1 2 3 4 5 6 7 8 9 10 11
  multi_iter_1dt iter1d(data);
  EXPECT_EQ(iter1d.stride(fluid::dim_x), 1);

  auto strided_iter = iter1d.as_strided_iterator(fluid::dim_x);

  EXPECT_EQ(*iter1d                        , 0);
  EXPECT_EQ(*iter1d.offset(1, fluid::dim_x), 1);
  EXPECT_EQ(*iter1d.offset(2, fluid::dim_x), 2);
  EXPECT_EQ(*iter1d.offset(3, fluid::dim_x), 3);
  EXPECT_EQ(*iter1d.offset(4, fluid::dim_x), 4);
  EXPECT_EQ(*iter1d.offset(5, fluid::dim_x), 5);
  EXPECT_EQ(*iter1d.offset(6, fluid::dim_x), 6);
  EXPECT_EQ(*iter1d.offset(7, fluid::dim_x), 7);

  EXPECT_EQ(*iter1d                        , *strided_iter);
  EXPECT_EQ(*iter1d.offset(1, fluid::dim_x), *strided_iter.offset(1));
  EXPECT_EQ(*iter1d.offset(2, fluid::dim_x), *strided_iter.offset(2));
  EXPECT_EQ(*iter1d.offset(3, fluid::dim_x), *strided_iter.offset(3));
  EXPECT_EQ(*iter1d.offset(4, fluid::dim_x), *strided_iter.offset(4));
  EXPECT_EQ(*iter1d.offset(5, fluid::dim_x), *strided_iter.offset(5));
  EXPECT_EQ(*iter1d.offset(6, fluid::dim_x), *strided_iter.offset(6));
  EXPECT_EQ(*iter1d.offset(7, fluid::dim_x), *strided_iter.offset(7));
}

TEST(strided_iter_host_tests, iterates_correctly_for_2d_data)
{
  using namespace fluid;

  // Create information to define 3 x 2 x 2 dimensional space
  using dim_info_2dt   = DimInfoCt<4, 3>;
  using multi_iter_2dt = MultidimIterator<std::size_t, dim_info_2dt>;

  constexpr auto size = dim_info_2dt().total_size();
  std::size_t data[size];
  for (const auto i : range(size)) { data[i] = i; }

  // Create iterator over the space, the data will look like:
  //       |- x x  x x -|
  //  ------------------------
  //    y  |  0 1  2  3
  //    y  |  4 5  6  7
  //    y  |  8 9 10 11
  multi_iter_2dt iter2d(data);

  EXPECT_EQ(iter2d.stride(fluid::dim_x), 1);
  EXPECT_EQ(iter2d.stride(fluid::dim_y), 4);

  // Test iteration over x:
  auto strided_iter_x = iter2d.as_strided_iterator(fluid::dim_x);
  EXPECT_EQ(*strided_iter_x      , 0);
  EXPECT_EQ(*(strided_iter_x + 1), 1);
  EXPECT_EQ(*(strided_iter_x + 2), 2);
  EXPECT_EQ(*(strided_iter_x + 3), 3);

  strided_iter_x = 
    iter2d.offset(1, fluid::dim_y).as_strided_iterator(fluid::dim_x);
  EXPECT_EQ(*strided_iter_x      , 4);
  EXPECT_EQ(*(strided_iter_x + 1), 5);
  EXPECT_EQ(*(strided_iter_x + 2), 6);
  EXPECT_EQ(*(strided_iter_x + 3), 7);

  strided_iter_x =
    iter2d.offset(2, fluid::dim_y).as_strided_iterator(fluid::dim_x);
  EXPECT_EQ(*strided_iter_x      ,  8);
  EXPECT_EQ(*(strided_iter_x + 1),  9);
  EXPECT_EQ(*(strided_iter_x + 2), 10);
  EXPECT_EQ(*(strided_iter_x + 3), 11);

  // Test iteration over y:
  auto strided_iter_y = iter2d.as_strided_iterator(fluid::dim_y);
  EXPECT_EQ(*strided_iter_y      , 0);
  EXPECT_EQ(*(strided_iter_y + 1), 4);
  EXPECT_EQ(*(strided_iter_y + 2), 8);

  strided_iter_y = 
    iter2d.offset(1, fluid::dim_x).as_strided_iterator(fluid::dim_y);
  EXPECT_EQ(*strided_iter_y      , 1);
  EXPECT_EQ(*(strided_iter_y + 1), 5);
  EXPECT_EQ(*(strided_iter_y + 2), 9);

  strided_iter_y = 
    iter2d.offset(2, fluid::dim_x).as_strided_iterator(fluid::dim_y);
  EXPECT_EQ(*strided_iter_y      ,  2);
  EXPECT_EQ(*(strided_iter_y + 1),  6);
  EXPECT_EQ(*(strided_iter_y + 2), 10);

  strided_iter_y = 
    iter2d.offset(3, fluid::dim_x).as_strided_iterator(fluid::dim_y);
  EXPECT_EQ(*strided_iter_y      ,  3);
  EXPECT_EQ(*(strided_iter_y + 1),  7);
  EXPECT_EQ(*(strided_iter_y + 2), 11);
}

int main(int argc, char** argv) 
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}