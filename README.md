# Learning Exchange Effects via Continuous Normalizing Flows

## Introduction

This a simulation code written as part of my doctoral project for attempting to mitigate the sign problem, present in electron systems. This project was built using C++ 11 and Python 3.13. Part of it uses some of the code present in the [PIMD Pro](https://github.com/xiongyunuo/pimd-pro-2) software package.

## Funcionality

The current funcionality includes:

- Calculation of the average sign dependence on an inferred backflow transformation via maximum likelihood estimation of simulation data using continous normalizing flows,
- Generation of exact input and backflow tranformation resulting data from an adjusted existing stochastic ring polymer path integral simulation code.

## Structure

There are two main structures:

- The `Processing` structure in ```src/processing/main.py``` which contains the transformation analysis software,
- The `Simulation` structure in ```src/simulation/``` which contains the adjusted path integral simulation code.

## Changelog

This program has had the following releases:


### 1.0 Stable - 27th of February 2025

- Initial release of the software.