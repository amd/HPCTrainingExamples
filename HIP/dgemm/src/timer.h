/***************************************************************************
 Copyright (c) 2022, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

#ifndef _TIMER_H_
#define _TIMER_H_
#include <chrono>
class timer{
public:
  timer() = default;

  void
  tick(){
     m_t0 = std::chrono::high_resolution_clock::now();
  }

  template <typename P>
  double
  tock(){
     return std::chrono::duration_cast<P>(
        std::chrono::high_resolution_clock::now() - m_t0
     ).count();
  }

private:
  using time_point = std::chrono::time_point<std::chrono::system_clock>;
  time_point m_t0;
  time_point m_t1;
};


#endif /* _TIMER_H_ */
