#include "random.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <iostream>
#include <cmath>

std::uniform_real_distribution<double> generator;
std::mt19937 algorithm;

void XSetRandSeed(XRandUInt seed)
{
  algorithm = std::mt19937(seed);

  generator = std::uniform_real_distribution<double>(0.0, 1.0);
}

XRandF XRandFloat()
{
  return generator(algorithm);
}

XRandF XRandGauss()
{
  static XRandF first, second;
  static bool has = false;
  if (has)
  {
    has = false;
    return second;
  }
  has = true;
  XRandF r = std::sqrt(-2 * std::log(XRandFloat()));
  XRandF angle = 2 * M_PI * XRandFloat();
  first = r * cos(angle);
  second = r * sin(angle);
  return first;
}