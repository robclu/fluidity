//==--- fluidity/tests/setting_tests.cpp ------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  setting_tests.cpp
/// \brief This file defines tests for settings.
//
//==------------------------------------------------------------------------==//

#include <fluidity/setting/option.hpp>
#include <fluidity/setting/option_holder.hpp>
#include <fluidity/setting/option_manager.hpp>
#include <fluidity/utility/type_traits.hpp>
#include <gtest/gtest.h>

using namespace fluid;

// Example base class struct to to create options for.
struct MockBase {
  // Runs the implementation.
  virtual void run(int& result) = 0;

  virtual ~MockBase() {}
};

template <std::size_t V> struct Dim {
  static constexpr auto value = V;
};

// Option for setting the dimensions:
struct DimOpt : public setting::Option<DimOpt> {
  static constexpr const char* type = "dimensions";
  static constexpr auto num_choices = std::size_t{3};

  static constexpr auto default_choice()
  {
    return setting::OptionHolder<Dim<1>>{"1D"};
  }
    
  /// Returns a tuple of the option choices.
  constexpr auto choice_list() const
  {
    return std::make_tuple(
      setting::OptionHolder<Dim<1>>{"1D"},
      setting::OptionHolder<Dim<2>>{"2D"}, 
      setting::OptionHolder<Dim<3>>{"3D"}
    );
  }
};

struct MethodA {
  template <typename D>
  int operator()(D&& dim) { return (D::value + 1) * 2; }
};

struct MethodB {
  template <typename D>
  int operator()(D&& dim) { return (D::value + 5) * 2; }
};

// Option for setting the method.
struct MethodOpt : public setting::Option<MethodOpt> {
  static constexpr const char* type = "method";
  static constexpr auto num_choices = std::size_t{2};

  static constexpr auto default_choice()
  {
    return setting::OptionHolder<MethodA>{"method_a"};
  }
    
  /// Returns a tuple of the option choices.
  constexpr auto choice_list() const
  {
    return std::make_tuple(
      setting::OptionHolder<MethodA>{"method_a"},
      setting::OptionHolder<MethodB>{"method_b"} 
    );
  }
};

template <typename... Ts>
struct MockImpl : public MockBase {
  // Defines the data type to use:
  using dim_t   = type_at_t<0, Dim<1>, Ts...>;
  // Defines the method to use:
  using method_t = type_at_t<1, MethodA, Ts...>;

  template <typename... Us>
  using make_type_t = MockImpl<Us...>;

  // Defines the option manager type:
  using op_manager_t = 
    setting::OptionManager<MockBase, MockImpl<>, DimOpt, MethodOpt>;

  // Runs implementation using the template types.
  void run(int& x) override 
  {
    x = method_t()(dim_t());
  }
};

using mock_manager_t = typename MockImpl<>::op_manager_t;

TEST(option_manager, can_create_default_type)
{
  mock_manager_t op_manager;

  int  x    = -1;
  auto impl = op_manager.create_default();
  impl->run(x);
  EXPECT_EQ(x, (1+1)*2);

  // Modify the setting but still create default, result should be the same.
  op_manager.set("dimensions", "2D");
  impl = op_manager.create_default();
  impl->run(x);
  EXPECT_EQ(x, (1+1)*2);
}

TEST(option_manager, can_create_from_option_settings)
{
  mock_manager_t op_manager;

  op_manager.set("dimensions", "2D").set("method", "method_b");
  int  x    = -1;
  auto impl = op_manager.create();
  impl->run(x);
  EXPECT_EQ(x, (2+5)*2);

  // Modify the setting but still create default, result should be the same.
  op_manager.set("dimensions", "3D").set("method", "method_a");
  impl = op_manager.create();
  impl->run(x);
  EXPECT_EQ(x, (3+1)*2);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
