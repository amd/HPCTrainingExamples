/***************************************************************************
 Copyright (c) 2022, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

#ifndef _UTILS_H_
#define _UTILS_H_


#include <string>
#include <vector>
#include <hip/hip_runtime.h>


using vstring = std::vector<std::string>;

/** Split input string by given delimiter
 *
 * @param str input string to split
 * @param del delimiter to split input string with
 */
vstring split(std::string const& str, char del);


/** Return local time representation in YYYY-mm-DD HH:MM:SS.zzz format
 *
 */
std::string now_str();



#ifdef DEBUG
inline hipError_t check_stat(hipError_t err){
   if (err == hipSuccess){
     return err;
   }
   fprintf(
      stderr,
      " %s: %s",
      hipGetErrorName(err),
      hipGetErrorString(err)
   );
   exit(1);
}
#else
inline hipError_t check_stat(hipError_t err){
   return err;
}
#endif


/** Insert input delimeter between vector elements
 *
 * @param vec input vector
 * @param del delimiter character
 * @return string compose of 'vec' elements separated by 'del'
 */
std::string
join(std::vector<std::string> const& vec, std::string const& del);


/** Return the maximum length of data in map
 *
 * @param data_map ordered or unordered data collection
 * @param accessor functor to access data individual collection data
 * @return maximum size of the data collection
 */
template <typename Map, typename F>
size_t
max_size(Map  const& data_map, F const& accessor){
   unsigned int max_size = 0;
   for (auto it=data_map.begin(); it!=data_map.end(); ++it){
     max_size = std::max<size_t>(accessor(it->second).size(), max_size);
   }
   return max_size;
}


/** Naive function to return file extension
 *
 * Don't want to add newer std as req
 *
 * @param path file path or filename
 * @return file extenison or empty string if none
 */
std::string
extension(std::string const& path);


/** Convert string to lower case
 *
 * @param str input string
 * @return string converted to lower case
 */
std::string
to_lower(std::string const& str);


struct Stats{
  double mean;
  double max;
  double min;
  double variance;
};



Stats
basic_stats(std::vector<double> const& data);


#endif /* _UTILS_H_ */
