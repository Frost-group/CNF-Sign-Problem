#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "particle.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <complex>

class XSimulation
{
private:
  XParticle *particles;
  XParticle center;
  int N, P;
  XNum beta, T;
  XNum vi;
  XNum beta2, vi2, g2, s2;
  XNum omgP;
  XNum omg0;
  XNum g, s;
  XNum **ENkCache;
  XNum *VBCache;
  std::complex<double> *VBCache2;
  XNum *ForceVBCache;
  XNum *ForceCache;
  XNum t, h;
  int count;

public:
  int step;
  int skip;
  bool ok;
  inline int Index(int l, int j) { return (l - 1) * P + j - 1; }
  XNum Distance(XParticle *p1, XParticle *p2);
  void RelativeDistance(XParticle *p1, XParticle *p2, XNum *displace);
  XSimulation() {}
  void Initial(std::istream &in, XNum seed);
  void Dump(std::ostream &observables_file, std::ostream &data_file, std::ostream &density_file);
  void VelocityRescale();
  XNum *Force();
  void UpdateMNHC_VV3();
  XNum ENk(int N2, int k);
  int NextIndex(int l, int j, int N2, int k);
  int PrevIndex(int l, int j, int N2, int k);
  XNum XExp(XNum k, XNum E, XNum EE);
  std::complex<double> XExp2(XNum k, std::complex<double> E, std::complex<double> EE);
  XNum XMinE(int N2);
  void FillVB();
  std::complex<double> XMinE2(int N2);
  void FillVB2();
  void dENk(int N2, int k, int l, int j, XNum *res);
  void FillForceVB();
  void FillENk();
  void TrapForce(int index, XNum *res);
  void PairForce(int index, int index2, XNum *res);
  XNum VBEnergy();
  XNum TrapEnergy(int index);
  XNum PairEnergy(int index, int index2);
  XNum TotalEnergy();
  void NHForce(XNum m, int f, XNum *v, XNum *Q, XNum *vtheta, XNum *res);
  XNum Temperature();
  void SetOmg(XNum o) { omg0 = o; }
  XNum Partition();
  std::complex<double> Partition2();
  ~XSimulation();

  XNum getJacobianElement(int untransformed_index, int transformed_index, int bead_index);
  XNum calculateDeterminant(XNum *matrix);
  XNum *getBackflowShift(XParticle *particle);
  XNum getBackflowAdjustment();

  // Define strength of backflows
  XNum strengths[1] = {-15.83};
  XNum scales[1] = {0.05374};

  // Control static particles
  int mobile_particle = -1;
  int static_seed = 1;
};

#endif
