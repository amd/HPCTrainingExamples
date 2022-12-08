/***************************************************************************
 Copyright (c) 2022, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

#ifndef _DARRAY_H_
#define _DARRAY_H_


#include <hip/hip_runtime.h>
#include "utils.h"



/** Device array
 *
 */
template <
  typename T,
  typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type
>
class darray{
public:

  /** Allocate an array for n number of elements
   */
  darray(size_t n){
     check_stat(
        hipMalloc((void**)&m_data, sizeof(T)*n)
     );
  };

  ~darray(){
     check_stat(
        hipFree(m_data)
     );
  };

  darray(darray<T> && that){
     m_data = std::move(that.m_data);
  };

  darray<T> const& operator=(darray<T> && that){
     m_data = std::move(that.data);
     return *this;
  }

  darray(darray<T> const&) = delete;
  darray<T> const& operator=(darray<T> const&) = delete;


  operator T*() const{
     return m_data;
  }


  T*
  data(){
     return m_data;
  }


private:
  T *m_data;
};


#endif /* _DARRAY_H_ */
