#include "simulation.hpp"
#include <iostream>
#include <fstream>
#include <ctime>

int main(int argc, char *argv[])
{
  std::ofstream out_file;
  std::ifstream in_file;
  XSimulation simul;

  out_file.open("output_" + std::string(argv[1]) + ".txt");
  in_file.open("options.txt");

  simul.Initial(in_file, atoi(argv[1]));

  clock_t t;
  int i;

  std::cout << "Starting simulation..." << std::endl;

  t = clock();

  for (i = 0; i < simul.step; ++i)
  {
    simul.PeriodBoundary();
    simul.UpdateMNHC_VV3();

    if (!simul.ok)
    {
      std::cout << "Simulation failed after " << i << " steps" << std::endl;
      return 0;
    }

    if (i % 1000 == 0)
      std::cout << "Completed " << i << " steps!" << std::endl;
  }

  std::cout << "Saving data..." << std::endl;

  simul.Dump(out_file);

  std::cout << "The total energy was calculated to be " << simul.TotalEnergy() << "!" << std::endl;

  t = clock() - t;

  std::cout << "Simulation finished in " << t / CLOCKS_PER_SEC << " seconds!" << std::endl;

  return 0;
}
