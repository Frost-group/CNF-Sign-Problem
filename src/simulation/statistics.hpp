#ifndef STATISTIC_X_HPP
#define STATISTIC_X_HPP

#include "particle.hpp"

class XStatistic {
public:
  XNum e;
  XNum T;
  XNum *den;
  int num;
  XStatistic();
  void Init(int size);
  ~XStatistic() { delete[] den; }
};

#endif