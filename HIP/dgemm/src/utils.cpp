/***************************************************************************
 Copyright (c) 2022, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

#include <sstream>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <chrono>
#include "utils.h"



vstring
split(std::string const& str, char del){
   vstring ret;
   std::stringstream itr(str);
   std::string val;
   while (std::getline(itr, val, del)){
     ret.push_back(val);
   }
   return ret;
}


std::string
now_str(){
   auto const now = std::chrono::high_resolution_clock::now();
   auto const now_t = std::chrono::system_clock::to_time_t(now);
   auto const ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
   std::stringstream ss;
   ss << std::put_time(std::localtime(&now_t), "%Y-%m-%d %H:%M:%S")
      << '.' << std::setfill('0') << std::setw(3) << ms.count();

   return ss.str();
}


std::string
join(std::vector<std::string> const& vec, std::string const& del){
   std::string ret;
   for (size_t i=0; i<vec.size(); ++i){
     if (i != 0){
       ret += ",";
     }
     ret += vec[i];
   }
   return ret;
}


std::string
extension(std::string const& path){
   auto const last_index = path.find_last_of(".");
   if (last_index >= path.size()){
     return "";
   }

   return path.substr(last_index + 1);
}


std::string
to_lower(std::string const& str){
   auto ret = str;
   std::transform(
      ret.begin(), ret.end(), ret.begin(),
      [](unsigned char x){ return std::tolower(x);});
   return ret;
}


Stats
basic_stats(std::vector<double> const& data){
  if (data.size() == 0){
    return Stats();
  }

  Stats ret;
  auto const mean = std::accumulate(data.begin(), data.end(), 0.0) / (double)data.size();;
  auto const min_max = std::minmax_element(data.begin(), data.end());

  ret.variance = std::accumulate(
     data.begin(), data.end(), 0.0, [mean](double x, double y){
        auto const m1 = y - mean;
        return x + m1*m1;
     }
  ) / (double)data.size();

  ret.mean = mean;
  ret.min = *min_max.first;
  ret.max = *min_max.second;
  return ret;
}
