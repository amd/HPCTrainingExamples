/***************************************************************************
 Copyright (c) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

#include "serialize.h"
#include "utils.h"




std::ostream&
elem_serialize(std::string const& d, std::ostream &stream){
   stream << "\"" + d + "\"";
   return stream;
}


std::ostream&
jserialize_str(std::vector<std::string> const& data, std::ostream &stream){
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


std::string
jserialize(vstring const& data){
  std::ostringstream stream;
  jserialize(data, stream);
  return stream.str();
}



std::string
jserialize(args const& inp){
   std::string ret = "{";

   ret += "\"m\":" + std::to_string(inp.m) + ",";
   ret += "\"n\":" + std::to_string(inp.n) + ",";
   ret += "\"k\":" + std::to_string(inp.k) + ",";
   ret += "\"i\":" + std::to_string(inp.iter_count) + ",";
   ret += "\"r\":" + std::to_string(inp.rep_count) + ",";
   ret += "\"o\":\"" + inp.output_fn + "\",";
   ret += "\"d\":" + jserialize(inp.device_ids);
   ret += "}";

   return ret;
}



void
serialize(
      std::unordered_map<int, dgemm_results> const& map,
      args const& inp,
      std::ostream &stream){

   std::string flop_vec;
   for (auto it=map.begin(); it!=map.end(); ++it){
     if (it != map.begin()){
       flop_vec += ",";
     }
     flop_vec += "\"" + std::to_string(it->first) + "\":"
       + jserialize(it->second.flops);
   }

   std::string time_vec;
   for (auto it=map.begin(); it!=map.end(); ++it){
     if (it != map.begin()){
       time_vec += ",";
     }
     time_vec += "\"" + std::to_string(it->first) + "\":"
       + jserialize(it->second.time_points);
   }

   stream << "{"
          << "\"flop_rates\": {" + flop_vec + "},"
          << "\"times\": {" + time_vec  + "},"
          << "\"args\":" + jserialize(inp)
          << "}";
}


void
serialize_csv(
      std::unordered_map<int, dgemm_results> const& map,
      args const& inp,
      std::ostream &stream){

   // Write headers
   std::vector<std::string> cols;
   for (auto it=map.begin(); it!=map.end(); ++it){
     auto const dev_id = it->first;
     auto const& data = it->second;
     cols.push_back("t_" + std::to_string(dev_id));
     cols.push_back("flops_" + std::to_string(dev_id));
   }
   auto const header_str = join(cols, ",");
   stream << header_str << "\n";

   auto const N = max_size(map, [](dgemm_results const& elem){
      return elem.flops; }
   );


   // Write data
   for (size_t i=0; i<N; ++i){
     std::vector<std::string> row;

     for (auto it=map.begin(); it!=map.end(); ++it){
       auto const dev_id = it->first;
       auto const& data = it->second;

       std::string t, x;
       if (i < data.time_points.size()){
         t = "\"" + data.time_points[i] + "\"";
         x = std::to_string(data.flops[i]);
       }

       row.push_back(t);
       row.push_back(x);
     }

     stream << join(row, ",") << "\n";
   }
}
