# Learning Exchange Effects via Continuous Normalizing Flows

## Introduction

This a simulation code written as part of my doctoral project for attempting to mitigate the sign problem, present in electron systems. This project was built using C++ 11 and Python 3.13. Part of it uses some of the code present in the [PIMD Pro](https://github.com/xiongyunuo/pimd-pro-2) software package and the [blackbox](https://github.com/paulknysh/blackbox) tool.

## Funcionality

The current funcionality includes:

- Calculation of the average sign dependence on an inferred backflow transformation via maximum likelihood estimation of simulation data using continous normalizing flows,
- Generation of exact input and backflow tranformation resulting data from an adjusted existing stochastic ring polymer path integral simulation code.

The resulting image files are labelled specifically to be compiled into a movie using the [ffmpeg](https://ffmpeg.org/ffmpeg.html) package via:

```
ffmpeg -f image2 -r 60 -pattern_type glob -i 'src/processing/output/strength_*.png' -vcodec libx264 -pix_fmt yuv420p strengths.mp4
```

## Structure

There are two main structures:

- The `Processing` structure in ```src/processing/``` which contains the backflow learning, nodal plotting and grid search software,
- The `Simulation` structure in ```src/simulation/``` which contains the adjusted path integral simulation code.

## Changelog

This program has had the following releases:


### 1.0 Stable - 27th of February 2025

- Initial release of the software.

### 2.0 Stable - 11th of April 2025

- Correct implementation of the backflow transformation in the simulation part of the software.
- Validated to learn the fermionic nodal surface with multiple backflow functions.
- Various analysis and usability improvements.

### 3.0 Stable - 9th of July 2025

- Added observable measurements, including for backflow parameter prediction.
- Added nodal surface data generation, processing and plotting.

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
