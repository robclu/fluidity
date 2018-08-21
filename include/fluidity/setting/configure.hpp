//==--- fluidity/setting/configure.hpp --------------------- -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  configure.hpp
/// \brief This file defines functionality to configure settings.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_SETTING_CONFIGURE_HPP
#define FLUIDITY_SETTING_CONFIGURE_HPP

#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace fluid   {
namespace setting {

auto tokenize(const std::string& str, char delim = ':')
{
  std::vector<std::string> result;
  int i = 0, s = 0;
  while (i < str.length())
  {
    s = i;
    while (str[i] != delim && i < str.length()) 
    {
      i++;
    }

    result.emplace_back(std::move(str.substr(s, i > 0 ? i - 1 : 0)));
    auto& s = result.back();
    s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
    i++;
  }
  return result;
}

/// Configures the \p manager from the \p file. This reads the file line by
/// line, tockenizing the lines into the option to set and the value to set the
/// option to. Each option must be on its own line with the form
/// \begin{code}
///   option : option_value
/// \end{code}
/// \param[in] manager The option manager to configure.
/// \param[in] file    The file to get the settings values from.
/// \tparam    Manager The type of the option manager.
template <typename Manager>
void configure_from_file(Manager& manager, std::string file)
{
  constexpr const char* comment = "#";
  std::string   line;
  std::ifstream f(file);

  if (!f.is_open())
  {
    printf("Failed to open the settings file: %s\n", file.c_str());
    return;
  }

  while (std::getline(f, line))
  {
    if (std::strcmp(&line[0], comment) == 0)
    {
      continue;
    }

    auto tokens = std::move(tokenize(line));
    for (const auto& token : tokens)
    { 
    }
  }
}

}} // namespace fluid::setting

#endif // FLUIDITY_SETTING_CONFIGURE_HPP