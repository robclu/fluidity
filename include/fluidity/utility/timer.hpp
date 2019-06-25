//==--- fluidity/utility/timer.hpp ------------------------- -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  timer.hpp
/// \brief This file defines a simple timer class.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_UTILITY_TIMER_HPP
#define FLUIDITY_UTILITY_TIMER_HPP

#include <chrono>

namespace fluid {
namespace util  {

/// The Timer class is a class which provides simple timing functionality.
/// Duration The type of the duration to get elapsed time for the timer.
template <typename Duration>
class Timer {
  /// Defines the type of the time point.
  using time_point_t = decltype(std::chrono::high_resolution_clock::now());

 public:
  /// Initializes the timer, starting it.
  Timer() : _start(std::chrono::high_resolution_clock::now()),
            _end(std::chrono::high_resolution_clock::now()) {}

  /// Starts or restarts the timer.
  void restart()
  {
    _start = std::chrono::high_resolution_clock::now();
  }

  /// Stops the timer.
  void stop()
  {
    _end = std::chrono::high_resolution_clock::now();
  }

  /// Returns the current elapsed time since the timer was started.
  auto elapsed_time()
  {
    stop();
    return std::chrono::duration_cast<Duration>(_end - _start).count();
  }

 private:
  time_point_t _start;  //!< Starting time of the timer.
  time_point_t _end;    //!< End time of the timer.
};

/// Defines the default timer for timing.
using default_timer_t = Timer<std::chrono::milliseconds>;

/// Defines the type of a high resolution timer.
using high_res_timer_t = Timer<std::chrono::nanoseconds>;

}} // namespace fluid::util

#endif // FLUIDITY_UTILITY_TIMER_HPP