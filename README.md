# Learning Exchange Effects via Continuous Normalizing Flows

## Introduction

This a simulation code written as part of my doctoral project for attempting to mitigate the sign problem, present in electron systems. This project was built using C++ 11 and Python 3.13.

## Funcionality

The current funcionality includes:

- Maximisation of the average sign via an applied data tranformation using a continous normalizing flows implementation,
- Generation of exact input data from an adjusted existing stochastic ring polymer path integral simulation code.

## Structure

There are two main structures:

- The `Processing` structure in ```src/processing/main.py``` which contains the transformation analysis software,
- The `Simulation` structure in ```src/simulation/``` which contains the adjusted path integral simulation code.

## Changelog

This program is in development and hasn't had any releases.