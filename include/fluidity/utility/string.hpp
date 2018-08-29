//==--- fluidity/utility/string.hpp ------------------------ -*- C++ -*- ---==//
//            
//                                Fluidity
// 
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  string.hpp
/// \brief This file defines string based utility functions.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_UTILITY_STRING_HPP
#define FLUIDITY_UTILITY_STRING_HPP

#include <algorithm>
#include <string>
#include <vector>

namespace fluid {
namespace util  {

/// Removes the \p character from a \p str.
/// \param[in] str        The string to remove the character from.
/// \param[in] character  The character to remove.
void remove(std::string& str, char character)
{
  str.erase(std::remove(str.begin(), str.end(), character), str.end());
}

/// Tokenizes the \p str string into tokens using the \p delim as the delimeter,
/// with the option to remove whitespace from the tokens. A container of tokens
/// is returned.
/// \param[in] str   The string to tokenize.
/// \param[in] delim The delimeter which separates tokens.
/// \param[in] clean If the tokens must be cleaned (whitespace removed).
auto tokenize(const std::string& str, char delim = ':', bool clean = true)
{
  std::vector<std::string> result;
  int i = 0, s = 0;
  while (i < str.length())
  {
    s = i;
    while (str[i] != delim && i < str.length()) { i++; }

    result.emplace_back(std::move(str.substr(s, i > 0 ? i - 1 : 0)));

    if (clean) { remove(result.back(), ' '); }
    i++;
  }
  return result;
}

}} // namespace fluid::util

#endif // FLUIDITY_UTILITY_STRING_HPP