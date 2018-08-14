//==--- fluidity/io/data_writer.hpp ------------------------ -*- C++ -*- ---==//
//
//                                Fluidity
//                                
//                      Copyright (c) 2018 Rob Clucas.
//
//  This file is distributed under the MIT License. See LICENSE for details.
//
//==------------------------------------------------------------------------==//
//
/// \file  data_writer.hpp
/// \brief This file defines the interface for data writing.
//
//==------------------------------------------------------------------------==//

#ifndef FLUIDITY_IO_DATA_WRITER_HPP
#define FLUIDITY_IO_DATA_WRITER_HPP

namespace fluidity {
namespace io       {

/// The DataOutputter class stores properties common to all outputters, as well
/// as defines the interface for outputting.
class DataOutputter {
 public:
  /// Sets the \p prefix and \p extension for data outputting.
  DataWriter(std::string prefix, std::string extension)
  : _prefix(prefix), _extension(extension) {}

 protected:
  std::string prefix    = "";       //!< The prefix of the output files.
  std::string extension = ".dat";   //!< The extension for the output files.
};

template <typename It, typename Transform = VoidTransform>
struct OutputInfo {
  /// Defines the type of the iterator to output the data from.
  using iter_t      = std::decay_t<It>;
  /// Defines the type of the transform to apply to elements before outputting.
  using transform_t = std::decay_t<Transform>;

  OutputInfo(const iter_t& iter) : _iter(iter) {}

 private:
  const iter_t& _iter;
};


/// The AsciiOutputter class outputs simulation data in a .dat file with the
/// first lines being comments lines (starting with #'s) and specifying the time
/// of the simulation and the data for each of the columns. The data for each
/// position in the simulation grid is specified on its own line. The output
/// file will look at follows:
///
/// \begin{code}
///  # t          : <simulation time for data output>
///  # Column 1   : dimension 0 position
///  # ...
///  # Column N   : dimension N position 
///  # Column N+1 : element 1 value at position <0,...,N> (e.g rho)
///  # ...
///  # Column N+M : element M value at position <0,...,M> (e.g v_y)
///  0.05 0.05 0.05 0.1 1.0 0.3 0.0
///  0.10 0.05 0.05 0.1 1.0 0.3 0.0
///  ...
///  ...
///  1.0 1.0 1.0 1.0 0.2 0.0 2.0
/// \end{code}
class AsciiOutputter {

};

/// The BlobOutputter class outputs simulation data as the raw simulation data.
/// For one dimension the data is a single row of data, as it is in the
/// simulation, for two dimensions the data is a matrix, also as is in the
/// simulation, while for three dimensions each file which is output is a 2D
/// slice of the 3D data along the z dimension, with the index of the slice in
/// the z dimension being appended to the filename. Thus, for a 10x10x10 grid in
/// 3D 10 files will be output with the indices 0-9 appended, and each file with
/// hold a 10x10 data matrix.
class BlobOutputter : public DataOutputter {
 public:
  using DataOutputter::DataOutputter;
 private:
};

}} // namespace fluidity::io

#endif // FLUIDITY_IO_DATA_WRITER_HPP