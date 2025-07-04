#include "simulation.hpp"
#include <iostream>
#include <fstream>
#include <ctime>

int main(int argc, char *argv[])
{
  std::ofstream observables_file;
  std::ofstream density_file;
  std::ofstream data_file;
  std::ifstream in_file;
  XSimulation simul;

  observables_file.open("output/observables_" + std::string(argv[1]) + ".csv");
  density_file.open("output/density_" + std::string(argv[1]) + ".csv");
  data_file.open("output/data_" + std::string(argv[1]) + ".csv");
  in_file.open("options.txt");

  simul.Initial(in_file, atoi(argv[1]));

  clock_t t;
  int i;

  std::cout << "Starting simulation..." << std::endl;

  t = clock();

  for (i = 0; i < simul.step; ++i)
  {
    simul.UpdateMNHC_VV3();

    if (!simul.ok)
    {
      std::cout << "Simulation failed after " << i << " steps" << std::endl;
      return 0;
    }

    if (i % simul.skip == 0)
    {
      std::cout << "Completed " << i << " steps!" << std::endl;
      simul.Dump(observables_file, data_file, density_file, false);
    }
  }

  std::cout << "Saving distributions..." << std::endl;

  simul.Dump(observables_file, data_file, density_file, true);

  observables_file.close();
  density_file.close();
  data_file.close();

  t = clock() - t;

  std::cout << "Simulation finished in " << t / CLOCKS_PER_SEC << " seconds!" << std::endl;

  return 0;
}
