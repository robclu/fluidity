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
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace fluid {
namespace util  {

/// Removes the \p character from a \p str.
/// \param[in] str        The string to remove the character from.
/// \param[in] character  The character to remove.
template <typename C, typename... Cs>
void remove(std::string& str, C&& character, Cs&&... chars)
{
  str.erase(std::remove(str.begin(), str.end(), character), str.end());
  auto v = std::vector<char>{static_cast<char>(chars)...};
  for (const auto c : v)
  {
    str.erase(std::remove(str.begin(), str.end(), c), str.end());
  }
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

/// Formats an output to an output stream for a name value pair, with an indent
/// and a width for the name. The following is the output format:
///
/// |--- indent ---||--- name ---||--- pad ---|: |--- value ---|
///
/// The number of characters before the ":" is ```indent + name_width```, and
/// the name is always left formatted.
/// \param[in] stream     The stream to output to.
/// \param[in] name       The name to output.
/// \param[in] value      The value to output.
/// \param[in] indent     The amount of indentation.
/// \param[in] name_width The width of the name.
/// \tparam    BufferN    The type of the name buffer.
/// \tparam    BufferV    The type of the value buffer.
template <typename BufferN, typename BufferV>
auto format_name_value(std::ostringstream& stream         ,
                       BufferN             name           ,
                       BufferV             value          ,
                       std::size_t         indent     = 0 , 
                       std::size_t         name_width = 12)
{ 
  stream << std::string(indent, ' ') << std::left << std::setw(name_width)
         << name << " : " << value << "\n";
}

}} // namespace fluid::util

#endif // FLUIDITY_UTILITY_STRING_HPP