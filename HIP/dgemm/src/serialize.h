/***************************************************************************
 Copyright (c) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

#ifndef SERIALIZE_H_
#define SERIALIZE_H_


#include <string>
#include <vector>
#include <ostream>
#include <unordered_map>
#include <sstream>
#include "utils.h"
#include "args.h"
#include "dgemm.h"



template <
  typename D,
  typename = std::enable_if_t<std::is_arithmetic<D>::value, D>
>
std::ostream&
elem_serialize(D const& d, std::ostream &stream){
   stream << std::to_string(d);
   return stream;
}


std::ostream&
elem_serialize(std::string const& d, std::ostream &stream);


template <typename V>
std::ostream&
jserialize(V const& data, std::ostream &stream){
  if (data.size() == 0){
    stream << "[]";
    return stream;
  }

  stream << "[";
  for (size_t i=0; i<data.size(); ++i){
    elem_serialize(data[i], stream);

    if (i != data.size() - 1){
      stream << ",";
    }
  }
  stream << "]";
  return stream;
}


std::ostream&
jserialize_str(std::vector<std::string> const& data, std::ostream &stream);


template <
  typename Vec,
  typename = std::enable_if_t<std::is_arithmetic<typename Vec::value_type>::value, Vec>
>
std::string
jserialize(Vec const& data){
  std::ostringstream stream;
  jserialize(data, stream);
  return stream.str();
}


std::string
jserialize(vstring const& data);


std::string
jserialize(args const& inp);


void
serialize(
      std::unordered_map<int, dgemm_results> const& map,
      args const& inp,
      std::ostream &stream);


void
serialize_csv(
      std::unordered_map<int, dgemm_results> const& map,
      args const& inp,
      std::ostream &stream);



#endif /* SERIALIZE_H_ */
