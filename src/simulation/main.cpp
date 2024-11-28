#include "simulation.hpp"
#include <fstream>
#include <iostream>
#include <ctime>

int main(int argc, char *argv[]) {

  std::ifstream in_file;
  in_file.open("options.txt");
  std::ofstream out_file;
  out_file.open("output.txt");
  XSimulation simul;
  /*simul.Initial(in_file);
  simul.Dump(out_file);
  simul.FillENk();
  simul.FillVB();
  simul.FillForceVB();
  simul.DumpForce(out_file);
  simul.DumpEnergy(out_file);
  simul.UpdateMNHC_VV3();
  simul.UpdateMNHC_VV3();
  //simul.Dump(out_file);
  for (int i = 0; i < 1000; ++i)
    simul.UpdateMNHC_VV3();
  simul.Dump(out_file);*/
  simul.Initial(in_file);
  int i;
  clock_t t;
  t = clock();
  for (i = 0; i < simul.skip; ++i) {
    simul.PeriodBoundary();
    simul.UpdateMNHC_VV3();
    if (!simul.ok) {
      simul.Dump(out_file);
      simul.DumpForce(out_file);
      simul.DumpEnergy(out_file);
      simul.NormalizeStat();
      simul.DumpStat(out_file);
      return 0;
    }
  }
  for (i = 0; i < simul.step; ++i) {
    simul.PeriodBoundary();
    simul.UpdateMNHC_VV3();
    if (!simul.ok) {
      simul.Dump(out_file);
      simul.DumpForce(out_file);
      simul.DumpEnergy(out_file);
      simul.NormalizeStat();
      simul.DumpStat(out_file);
      return 0;
    }
    simul.MakeStat();
    if (i % 100000 == 0)
      out_file << "i=" << i << std::endl;
  }
  //simul.Dump(out_file);
  simul.NormalizeStat();
  simul.DumpStat(out_file);
  t = clock() - t;
  out_file << (int)(((double)1000 * t) / CLOCKS_PER_SEC) << std::endl;
  return 0;
}