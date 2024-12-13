#ifndef RANDOM_HPP
#define RANDOM_HPP

#include <random>

typedef unsigned int XRandUInt;
typedef float XRandF;

extern std::uniform_real_distribution<double> generator;
extern std::mt19937 algorithm;

void XSetRandSeed(XRandUInt seed);
XRandF XRandFloat();
XRandF XRandGauss();

#endif