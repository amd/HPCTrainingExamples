/***************************************************************************
 Copyright (c) 2022, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

#ifndef _MATRIX_H_
#define _MATRIX_H_


#include <vector>


template <typename T>
class matrix{
public:
  matrix(int m, int n, T val=0)
    : m_rowcount(m), m_colcount(n)
  {
     m_data = std::vector<T>(m*n, val);
  }

  double const&
  operator()(int i, int j) const{
     return m_data[i*m_colcount + j];
  }

  double&
  operator()(int i, int j){
     return m_data[i*m_colcount + j];
  }

  int
  row_count() const noexcept{
     return m_rowcount;
  }

  int
  col_count() const noexcept{
     return m_colcount;
  }

  std::vector<T> const&
  data() const{
     return m_data;
  }

private:
  unsigned int m_rowcount = 0;
  unsigned int m_colcount = 0;
  std::vector<T> m_data;
};


using matrixd = matrix<double>;


#endif /* _MATRIX_H_ */
