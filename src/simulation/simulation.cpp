#include "simulation.hpp"
#include "random.hpp"

XNum XSimulation::Distance(XParticle *p1, XParticle *p2)
{
  XNum *p1_shift = getBackflowShift(p1);
  XNum *p2_shift = getBackflowShift(p2);

  XNum res = 0;
  int i;
  for (i = 0; i < D; ++i)
  {
    XNum a = (p1->coor[i] - p2->coor[i]) + (p1_shift[i] - p2_shift[i]);
    res += a * a;
  }

  delete[] p1_shift;
  delete[] p2_shift;

  return res;
}

void XSimulation::RelativeDistance(XParticle *p1, XParticle *p2, XNum *displace)
{
  XNum *p1_shift = getBackflowShift(p1);
  XNum *p2_shift = getBackflowShift(p2);

  int i;
  for (i = 0; i < D; ++i)
    displace[i] = (p1->coor[i] - p2->coor[i]) + (p1_shift[i] - p2_shift[i]);

  delete[] p1_shift;
  delete[] p2_shift;
}

void XSimulation::Initial(std::istream &in, XNum seed)
{
  XSetRandSeed(seed);

  std::string parameter;
  while (in >> parameter)
  {
    if (parameter == "N")
      in >> N;
    else if (parameter == "P")
      in >> P;
    else if (parameter == "T0")
    {
      double T0;
      in >> T0;

      T = T0;

      beta2 = 1 / (kB * T);
      beta = 1 / (kB * T);
    }
    else if (parameter == "vi")
      in >> vi;
    else if (parameter == "vi2")
      in >> vi2;
    else if (parameter == "omg0")
      in >> omg0;
    else if (parameter == "g")
    {
      in >> g;
      g2 = g;
    }
    else if (parameter == "s")
    {
      in >> s;
      s2 = s;
    }
    else if (parameter == "h")
      in >> h;
    else if (parameter == "step")
      in >> step;
    else if (parameter == "skip")
      in >> skip;
  }
  particles = new XParticle[N * P];
  int l, j;
  XNum vs[D];
  int i;
  for (i = 0; i < D; ++i)
    vs[i] = 0;
  for (l = 1; l <= N; ++l)
    for (j = 1; j <= P; ++j)
    {
      int index = Index(l, j);
      particles[index].m = 1;
      particles[index].n = l;
      particles[index].p = j;
      for (i = 0; i < D; ++i)
      {
        XNum tmp = XRandGauss() * std::sqrt(1 / (particles[index].m * beta));
        vs[i] += tmp;
        particles[index].vel[i] = tmp;
      }
      for (i = 0; i < D; ++i)
        particles[index].coor[i] = 1.0 * (XRandFloat() - 0.5);
    }
  for (i = 0; i < D; ++i)
    vs[i] /= N * P;
  for (l = 1; l <= N; ++l)
    for (j = 1; j <= P; ++j)
    {
      int index = Index(l, j);
      for (i = 0; i < D; ++i)
        particles[index].vel[i] -= vs[i];
      int j;
      for (i = 0; i < D; ++i)
        for (j = 0; j < M; ++j)
        {
          particles[index].theta[i][j] = 1;
          particles[index].vtheta[i][j] = 1;
          particles[index].Q[i][j] = 1.0;
        }
    }
  VelocityRescale();
  omgP = std::sqrt(P) / (beta * hBar);
  ENkCache = new XNum *[N];
  for (l = 1; l <= N; ++l)
    ENkCache[l - 1] = new XNum[l];
  VBCache = new XNum[N + 1];
  VBCache2 = new std::complex<double>[N + 1];
  ForceVBCache = new XNum[D * N * P];
  for (i = 0; i < D; ++i)
    center.coor[i] = 0.0;
  t = 0;
  ForceCache = NULL;
  count = 0;
  ok = true;

  // Set static random positions for the not mobile particles
  if (mobile_particle >= 0)
  {
    // Choose the single mobile particle
    mobile_particle = std::floor(XRandFloat() * N);

    // Set the immobile particle coordinate sames across all beads
    for (l = 0; l < N; ++l)
      if (l != mobile_particle)
      {
        XNum *position = new XNum[D];

        std::cout << "Fixed position for particle " << l << " at [";

        for (i = 0; i < D; ++i)
        {
          position[i] = 1.0 * (XRandFloat() - 0.5);

          if (i > 0)
            std ::cout << ", ";

          std::cout << position[i];
        }

        std::cout << "]." << std::endl;

        for (j = 0; j < P; ++j)
          for (i = 0; i < D; ++i)
            particles[l * P + j].coor[i] = position[i];

        delete[] position;
      }
  }

  // For grid search cross-check
  // scales[0] = 0.01 + 0.09 * XRandFloat();
  // strengths[0] = 30.0 * XRandFloat();
}

void XSimulation::Dump(std::ostream &observables_file, std::ostream &data_file, std::ostream &density_file)
{
  std::complex<double> sign = std::exp(Partition2() - Partition());
  XNum backflow_adjustment = getBackflowAdjustment();
  XNum temperature = Temperature();

  std::complex<double> energy = TotalEnergy() * sign * backflow_adjustment;
  std::complex<double> factor = sign * backflow_adjustment;

  int l, j;

  // Save data for the machine learning algorithm and the density
  for (j = 1; j <= P; ++j)
    for (l = 1; l <= N; ++l)
    {
      int index = Index(l, j);

      XNum *shift = getBackflowShift(&particles[index]);

      data_file << temperature << ", " << l << ", " << j << ", " << factor.real() << ", " << particles[index].coor[0] + shift[0] << ", " << particles[index].coor[1] + shift[1] << ", " << 0.0 << ", " << backflow_adjustment << std::endl;

      if (l - 1 == mobile_particle && mobile_particle >= 0)
        density_file << particles[index].coor[0] + shift[0] << ", " << particles[index].coor[1] + shift[1] << ", " << factor.real() << ", " << backflow_adjustment << std::endl;

      delete[] shift;
    }

  // Calculate any additional observables
  XNum optimal_backflow_value = 0.0;

  for (l = 0; l < N * P; ++l)
  {
    XNum *l_shift = getBackflowShift(&particles[l]);

    for (j = 0; j <= l; ++j)
    {
      XNum *j_shift = getBackflowShift(&particles[j]);

      optimal_backflow_value += (2 * pow(pow(particles[l].coor[0] + l_shift[0], 2) + pow(particles[l].coor[1] + l_shift[1], 2), 1.5) + 2 * pow(pow(particles[l].coor[0] + l_shift[0], 2) + pow(particles[j].coor[1] + j_shift[1], 2), 1.5) + (pow(particles[l].coor[0] + l_shift[0] - particles[j].coor[0] - j_shift[0], 2) + pow(particles[l].coor[1] + l_shift[1] - particles[j].coor[1] - j_shift[1], 2)) * pow(pow(particles[j].coor[0] + j_shift[0] + particles[l].coor[0] + l_shift[0], 2) + pow(particles[j].coor[1] + j_shift[1] + particles[l].coor[1] + l_shift[1], 2), 0.5)) / pow(pow(particles[l].coor[0] + l_shift[0], 2) + pow(particles[l].coor[1] + l_shift[1], 2), 2) / pow(pow(particles[j].coor[0] + j_shift[0], 2) + pow(particles[j].coor[1] + j_shift[1], 2), 2);

      delete[] j_shift;
    }

    delete[] l_shift;
  }

  optimal_backflow_value = temperature / optimal_backflow_value;

  observables_file << energy.real() << ", " << factor.real() << ", " << sign.real() << ", " << optimal_backflow_value << std::endl;
}

XSimulation::~XSimulation()
{
  delete[] particles;
  int l;
  for (l = 1; l <= N; ++l)
    delete[] ENkCache[l - 1];
  delete[] ENkCache;
  delete[] VBCache;
  delete[] VBCache2;
  delete[] ForceVBCache;
  if (ForceCache)
    delete[] ForceCache;
}

void XSimulation::VelocityRescale()
{
  XNum sum = 0;
  int l, j;
  for (l = 1; l <= N; ++l)
    for (j = 1; j <= P; ++j)
    {
      int index = Index(l, j);
      int i;
      for (i = 0; i < D; ++i)
        sum += particles[index].m * particles[index].vel[i] * particles[index].vel[i];
    }
  XNum lam = std::sqrt((N * P) * D * T / sum);
  for (l = 1; l <= N; ++l)
    for (j = 1; j <= P; ++j)
    {
      int index = Index(l, j);
      int i;
      for (i = 0; i < D; ++i)
        particles[index].vel[i] *= lam;
    }
}

XNum XSimulation::ENk(int N2, int k)
{
  XNum res = 0;
  int l, j;
  for (l = N2 - k + 1; l <= N2; ++l)
    for (j = 1; j <= P; ++j)
    {
      int index = Index(l, j);
      int index2 = NextIndex(l, j, N2, k);
      res += 0.5 * particles[index].m * omgP * omgP * Distance(&particles[index], &particles[index2]);
    }
  return res;
}

int XSimulation::NextIndex(int l, int j, int N2, int k)
{
  int res = Index(l, j + 1);
  if (j == P)
  {
    if (l == N2)
      res = Index(N2 - k + 1, 1);
    else
      res = Index(l + 1, 1);
  }
  return res;
}

int XSimulation::PrevIndex(int l, int j, int N2, int k)
{
  int res = Index(l, j - 1);
  if (j == 1)
  {
    if (l == N2 - k + 1)
      res = Index(N2, P);
    else
      res = Index(l - 1, P);
  }
  return res;
}

XNum XSimulation::XExp(XNum k, XNum E, XNum EE)
{
  if (vi == 0)
    return std::exp(-beta * E + EE);
  else
    return std::exp((k - 1) * std::log(vi) - beta * E + EE);
}

XNum XSimulation::XMinE(int N2)
{
  if (vi == 0)
    return beta * (ENkCache[N2 - 1][0] + VBCache[N2 - 1]);
  int k;
  XNum res = 100000000;
  for (k = 1; k <= N2; ++k)
  {
    XNum tmp = -(k - 1) * std::log(vi) + beta * (ENkCache[N2 - 1][k - 1] + VBCache[N2 - k]);
    if (tmp < res)
      res = tmp;
  }
  return res;
}

void XSimulation::FillVB()
{
  int N2, k;
  VBCache[0] = 0;
  for (N2 = 1; N2 <= N; ++N2)
  {
    XNum sum = 0;
    XNum tmp = XMinE(N2);
    for (k = 1; k <= N2; ++k)
    {
      if (vi == 0 && k - 1 != 0)
        continue;
      sum += XExp(k, ENkCache[N2 - 1][k - 1] + VBCache[N2 - k], tmp);
    }
    VBCache[N2] = (tmp - std::log(sum) + std::log(N2)) / beta;
  }
}

void XSimulation::dENk(int N2, int k, int l, int j, XNum *res)
{
  int i;
  for (i = 0; i < D; ++i)
    res[i] = 0;
  if (l >= N2 - k + 1 && l <= N2)
  {
    int index = Index(l, j);
    int index2 = NextIndex(l, j, N2, k);
    int index3 = PrevIndex(l, j, N2, k);
    XNum displace[D];
    RelativeDistance(&particles[index], &particles[index2], displace);
    for (i = 0; i < D; ++i)
      res[i] += particles[index].m * omgP * omgP * displace[i];
    RelativeDistance(&particles[index], &particles[index3], displace);
    for (i = 0; i < D; ++i)
      res[i] += particles[index].m * omgP * omgP * displace[i];
  }
}

void XSimulation::FillForceVB()
{
  int N2, k, l, j;
  XNum *res = new XNum[D * (N + 1)];
  XNum grad[D];
  for (l = 1; l <= N; ++l)
    for (j = 1; j <= P; ++j)
    {
      int i;
      for (i = 0; i < D; ++i)
        res[i] = 0;
      for (N2 = 1; N2 <= N; ++N2)
      {
        XNum sum2 = 0;
        XNum tmp = XMinE(N2);
        for (k = 1; k <= N2; ++k)
        {
          if (vi == 0 && k - 1 != 0)
            continue;
          sum2 += XExp(k, ENkCache[N2 - 1][k - 1] + VBCache[N2 - k], tmp);
        }
        for (i = 0; i < D; ++i)
        {
          XNum sum = 0;
          for (k = 1; k <= N2; ++k)
          {
            if (vi == 0 && k - 1 != 0)
              continue;
            dENk(N2, k, l, j, grad);
            sum += (grad[i] + res[D * (N2 - k) + i]) * XExp(k, ENkCache[N2 - 1][k - 1] + VBCache[N2 - k], tmp);
          }
          res[D * N2 + i] = sum / sum2;
        }
      }
      int index = Index(l, j);
      for (i = 0; i < D; ++i)
        ForceVBCache[D * index + i] = res[D * N + i];
    }
  delete[] res;
}

void XSimulation::FillENk()
{
  int l, j;
  for (l = 1; l <= N; ++l)
  {
    for (j = 1; j <= l; ++j)
      ENkCache[l - 1][j - 1] = ENk(l, j);
  }
}

void XSimulation::TrapForce(int index, XNum *res)
{
  int i;
  XNum displace[D];
  RelativeDistance(&particles[index], &center, displace);
  for (i = 0; i < D; ++i)
    res[i] = -particles[index].m * omg0 * omg0 * displace[i];
}

void XSimulation::PairForce(int index, int index2, XNum *res)
{
  int i;
  XNum displace[D];
  // XNum inter = (g/(M_PI*s*s))*std::exp(-Distance(&particles[index], &particles[index2])/(s*s));
  XNum inter = g / std::pow(Distance(&particles[index], &particles[index2]), 1.5);
  RelativeDistance(&particles[index], &particles[index2], displace);
  for (i = 0; i < D; ++i)
    res[i] = displace[i] * inter;
  // res[i] = ((2*displace[i])/(s*s))*inter;
}

XNum *XSimulation::Force()
{
  XNum *res = new XNum[D * N * P];
  FillENk();
  FillVB();
  FillForceVB();
  int l, j, k;
  for (l = 1; l <= N; ++l)
    for (j = 1; j <= P; ++j)
    {
      int index = Index(l, j);
      int i;
      for (i = 0; i < D; ++i)
        res[D * index + i] = -ForceVBCache[D * index + i] / particles[index].m;
    }
  XNum trap[D];
  for (l = 1; l <= N; ++l)
    for (j = 1; j <= P; ++j)
    {
      int index = Index(l, j);
      TrapForce(index, trap);
      int i;
      for (i = 0; i < D; ++i)
        res[D * index + i] += trap[i] / particles[index].m / P;
    }
  XNum inter[D];
  for (l = 1; l <= N; ++l)
    for (j = 1; j <= P; ++j)
    {
      int index = Index(l, j);
      for (k = 1; k <= N; ++k)
      {
        if (l == k)
          continue;
        int index2 = Index(k, j);
        PairForce(index, index2, inter);
        int i;
        for (i = 0; i < D; ++i)
          res[D * index + i] += inter[i] / particles[index].m / P;
      }
    }
  return res;
}

XNum XSimulation::VBEnergy()
{
  XNum *res = new XNum[N + 1];
  res[0] = 0;
  int N2, k;
  for (N2 = 1; N2 <= N; ++N2)
  {
    XNum tmp = XMinE(N2);
    XNum sum2 = 0;
    for (k = 1; k <= N2; ++k)
    {
      if (vi == 0 && k - 1 != 0)
        continue;
      sum2 += XExp(k, ENkCache[N2 - 1][k - 1] + VBCache[N2 - k], tmp);
    }
    XNum sum = 0;
    for (k = 1; k <= N2; ++k)
    {
      if (vi == 0 && k - 1 != 0)
        continue;
      sum += (res[N2 - k] - ENkCache[N2 - 1][k - 1]) * XExp(k, ENkCache[N2 - 1][k - 1] + VBCache[N2 - k], tmp);
    }
    res[N2] = sum / sum2;
  }
  XNum e = res[N];
  delete[] res;
  return e;
}

XNum XSimulation::TrapEnergy(int index)
{
  return 0.5 * particles[index].m * omg0 * omg0 * Distance(&particles[index], &center);
}

XNum XSimulation::PairEnergy(int index, int index2)
{
  // XNum inter = (g/(M_PI*s*s))*std::exp(-Distance(&particles[index], &particles[index2])/(s*s));
  XNum inter = g / std::sqrt(Distance(&particles[index], &particles[index2]));
  return 0.5 * inter;
}

XNum XSimulation::TotalEnergy()
{
  XNum e = 0;
  e += P * D * N / (2 * beta);
  e += VBEnergy();
  int l, j, k;
  for (l = 1; l <= N; ++l)
    for (j = 1; j <= P; ++j)
    {
      int index = Index(l, j);
      e += TrapEnergy(index) / P;
    }
  for (l = 1; l <= N; ++l)
    for (j = 1; j <= P; ++j)
    {
      int index = Index(l, j);
      for (k = 1; k <= N; ++k)
      {
        if (l == k)
          continue;
        int index2 = Index(k, j);
        e += PairEnergy(index, index2) / P;
      }
    }
  return e;
}

void XSimulation::NHForce(XNum m, int f, XNum *v, XNum *Q, XNum *vtheta, XNum *res)
{
  XNum sum = 0;
  int i;
  for (i = 0; i < f; ++i)
    sum += m * v[i] * v[i];
  res[0] = (sum - f * (1 / beta)) / Q[0];
  for (i = 1; i < M; ++i)
    res[i] = (Q[i - 1] * vtheta[i - 1] * vtheta[i - 1] - (1 / beta)) / Q[i];
}

void XSimulation::UpdateMNHC_VV3()
{
  if (ForceCache == NULL)
    ForceCache = Force();
  int j;
  XNum NHF[M];
  XNum v[1];
  for (j = 0; j < N * P; ++j)
  {
    // Only allow one particle to move
    if (std::floor(j / P) != mobile_particle && mobile_particle >= 0)
      continue;

    int i;
    for (i = 0; i < D; ++i)
    {
      if (IsNan(ForceCache[D * j + i]) || IsInf(ForceCache[D * j + i]))
      {
        ok = false;
        return;
      }
      v[0] = particles[j].vel[i];
      NHForce(particles[j].m, 1, v, particles[j].Q[i], particles[j].vtheta[i], NHF);
      particles[j].vel[i] = particles[j].vel[i] * std::exp(-0.5 * h * particles[j].vtheta[i][0]) + 0.5 * h * ForceCache[D * j + i] * std::exp(-0.25 * h * particles[j].vtheta[i][0]);
      int M2 = M / 2;
      int k;
      for (k = 1; k <= M2; ++k)
        particles[j].theta[i][2 * k - 2] = particles[j].theta[i][2 * k - 2] + h * particles[j].vtheta[i][2 * k - 2] / 2;
      for (k = 1; k <= M2; ++k)
        particles[j].vtheta[i][2 * k - 1] = particles[j].vtheta[i][2 * k - 1] * std::exp(-0.5 * h * ((k == M2) ? 0 : particles[j].vtheta[i][2 * k])) + 0.5 * h * NHF[2 * k - 1] * std::exp(-0.25 * h * ((k == M2) ? 0 : particles[j].vtheta[i][2 * k]));
      particles[j].coor[i] = particles[j].coor[i] + h * particles[j].vel[i];
      for (k = 1; k <= M2; ++k)
        particles[j].theta[i][2 * k - 1] = particles[j].theta[i][2 * k - 1] + h * particles[j].vtheta[i][2 * k - 1];
      v[0] = particles[j].vel[i];
      NHForce(particles[j].m, 1, v, particles[j].Q[i], particles[j].vtheta[i], NHF);
      for (k = 1; k <= M2; ++k)
        particles[j].vtheta[i][2 * k - 2] = particles[j].vtheta[i][2 * k - 2] * std::exp(-h * particles[j].vtheta[i][2 * k - 1]) + h * NHF[2 * k - 2] * std::exp(-0.5 * h * particles[j].vtheta[i][2 * k - 1]);
    }
  }
  t += h;
  delete[] ForceCache;
  ForceCache = Force();
  for (j = 0; j < N * P; ++j)
  {
    // Only allow one particle to move
    if (std::floor(j / P) != mobile_particle && mobile_particle >= 0)
      continue;

    int i;
    for (i = 0; i < D; ++i)
    {
      if (IsNan(ForceCache[D * j + i]) || IsInf(ForceCache[D * j + i]))
      {
        ok = false;
        return;
      }
      int M2 = M / 2;
      particles[j].vel[i] = particles[j].vel[i] * std::exp(-0.5 * h * particles[j].vtheta[i][0]) + 0.5 * h * ForceCache[D * j + i] * std::exp(-0.25 * h * particles[j].vtheta[i][0]);
      int k;
      for (k = 1; k <= M2; ++k)
        particles[j].theta[i][2 * k - 2] = particles[j].theta[i][2 * k - 2] + h * particles[j].vtheta[i][2 * k - 2] / 2;
      v[0] = particles[j].vel[i];
      NHForce(particles[j].m, 1, v, particles[j].Q[i], particles[j].vtheta[i], NHF);
      for (k = 1; k <= M2; ++k)
        particles[j].vtheta[i][2 * k - 1] = particles[j].vtheta[i][2 * k - 1] * std::exp(-0.5 * h * ((k == M2) ? 0 : particles[j].vtheta[i][2 * k])) + 0.5 * h * NHF[2 * k - 1] * std::exp(-0.25 * h * ((k == M2) ? 0 : particles[j].vtheta[i][2 * k]));
    }
  }
}

XNum XSimulation::Temperature()
{
  XNum sum = 0;
  int j;
  for (j = 0; j < N * P; ++j)
  {
    int i;
    for (i = 0; i < D; ++i)
      sum += particles[j].m * particles[j].vel[i] * particles[j].vel[i];
  }
  return sum / (D * N * P);
}

std::complex<double> XSimulation::XExp2(XNum k, std::complex<double> E, std::complex<double> EE)
{
  if (vi == 0)
    return std::exp(-beta * E + EE);
  else
    return std::exp((k - 1) * std::log(std::complex<double>(vi)) - beta * E + EE);
}

std::complex<double> XSimulation::XMinE2(int N2)
{
  if (vi == 0)
    return beta * (ENkCache[N2 - 1][0] + VBCache2[N2 - 1]);
  int k;
  std::complex<double> res = 100000000;
  for (k = 1; k <= N2; ++k)
  {
    std::complex<double> tmp = -XNum(k - 1) * std::log(std::complex<double>(vi)) + beta * (ENkCache[N2 - 1][k - 1] + VBCache2[N2 - k]);
    if (tmp.real() < res.real())
      res = tmp;
  }
  return res;
}

void XSimulation::FillVB2()
{
  int N2, k;
  VBCache2[0] = 0;
  for (N2 = 1; N2 <= N; ++N2)
  {
    std::complex<double> sum = 0;
    std::complex<double> tmp = XMinE2(N2);
    for (k = 1; k <= N2; ++k)
    {
      if (vi == 0 && k - 1 != 0)
        continue;
      sum += XExp2(k, ENkCache[N2 - 1][k - 1] + VBCache2[N2 - k], tmp);
    }
    VBCache2[N2] = (tmp - std::log(sum) + std::log(N2)) / beta;
  }
}

XNum XSimulation::Partition()
{
  XNum res = 0;
  res += VBCache[N];
  XNum e = 0;
  int l, j, k;
  for (l = 1; l <= N; ++l)
    for (j = 1; j <= P; ++j)
    {
      int index = Index(l, j);
      e += TrapEnergy(index) / P;
    }
  for (l = 1; l <= N; ++l)
    for (j = 1; j <= P; ++j)
    {
      int index = Index(l, j);
      for (k = 1; k <= N; ++k)
      {
        if (l == k)
          continue;
        int index2 = Index(k, j);
        e += PairEnergy(index, index2) / P;
      }
    }
  res += e;
  return -beta * res;
}

std::complex<double> XSimulation::Partition2()
{
  XNum tmpb = beta;
  XNum tmpv = vi;
  XNum tmpg = g;
  XNum tmps = s;
  beta = beta2;
  vi = vi2;
  g = g2;
  s = s2;
  int l, j;
  XNum omgP2 = std::sqrt(P) / (beta * hBar);
  for (l = 1; l <= N; ++l)
  {
    for (j = 1; j <= l; ++j)
      ENkCache[l - 1][j - 1] *= (omgP2 * omgP2) / (omgP * omgP);
  }
  FillVB2();
  std::complex<double> res = 0;
  res += VBCache2[N];
  XNum e = 0;
  int k;
  for (l = 1; l <= N; ++l)
    for (j = 1; j <= P; ++j)
    {
      int index = Index(l, j);
      e += TrapEnergy(index) / P;
    }
  for (l = 1; l <= N; ++l)
    for (j = 1; j <= P; ++j)
    {
      int index = Index(l, j);
      for (k = 1; k <= N; ++k)
      {
        if (l == k)
          continue;
        int index2 = Index(k, j);
        e += PairEnergy(index, index2) / P;
      }
    }
  res += e;
  for (l = 1; l <= N; ++l)
  {
    for (j = 1; j <= l; ++j)
      ENkCache[l - 1][j - 1] *= (omgP * omgP) / (omgP2 * omgP2);
  }
  beta = tmpb;
  vi = tmpv;
  g = tmpg;
  s = tmps;
  return -beta * res;
}

// Calculate the backflow shift
XNum *XSimulation::getBackflowShift(XParticle *particle)
{
  XNum *shift = new XNum[D];

  int dimension;
  for (dimension = 0; dimension < D; ++dimension)
    shift[dimension] = 0.0;

  // Make no backflow simulations more efficient
  if (sizeof(strengths) / sizeof(strengths[0]) == 1 && strengths[0] == 0.0)
    return shift;

  std::cout << "A";

  // Only allow backflows to apply at the same imaginary time step
  int particle_index;
  for (particle_index = 1; particle_index <= N; ++particle_index)
  {
    XParticle other_particle = particles[Index(particle_index, particle->p)];

    // Calculate distance
    double distance = 0.0;

    for (dimension = 0; dimension < D; ++dimension)
      distance += pow((particle->coor[dimension] - other_particle.coor[dimension]), 2);

    distance = sqrt(distance);

    // Apply backflows
    int backflow;
    for (backflow = 0; backflow < sizeof(strengths) / sizeof(strengths[0]); ++backflow)
      for (dimension = 0; dimension < D; ++dimension)
        shift[dimension] += (particle->coor[dimension] - other_particle.coor[dimension]) * strengths[backflow] / (1.0 + pow(distance / scales[backflow], 3));
  }

  return shift;
}

XNum XSimulation::getBackflowAdjustment()
{
  // Make no backflow simulations more efficient
  if (sizeof(strengths) / sizeof(strengths[0]) == 1 && strengths[0] == 0.0)
    return 1.0;

  XNum *jacobian = new XNum[N * N];
  XNum factor = 0.0;

  // Only particle coordinate at the same beed index interact
  for (int bead_index = 0; bead_index < P; bead_index++)
  {
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
        jacobian[i * N + j] = getJacobianElement(i, j, bead_index);

    factor += calculateDeterminant(jacobian);
  }

  delete[] jacobian;

  return factor;
}

XNum XSimulation::getJacobianElement(int untransformed_index, int transformed_index, int bead_index)
{
  XParticle *particle = &particles[Index(transformed_index, bead_index)];
  XNum element = 0.0;

  // Only allow backflows to apply at the same imaginary time step
  int particle_index;
  for (particle_index = 1; particle_index <= N; ++particle_index)
  {
    // Non-diagonal case
    if (untransformed_index == transformed_index)
      particle_index = transformed_index;

    XParticle other_particle = particles[Index(particle_index, bead_index)];

    // Calculate distance
    double distance = 0.0;

    for (int dimension = 0; dimension < D; ++dimension)
      distance += pow(particle->coor[dimension] - other_particle.coor[dimension], 2);

    distance = sqrt(distance);

    // Apply backflows
    int backflow;
    for (backflow = 0; backflow < sizeof(strengths) / sizeof(strengths[0]); ++backflow)
      if (untransformed_index == transformed_index)
        element += strengths[backflow] * (pow(distance / scales[backflow], 6) + 2.0) / pow(1.0 + pow(distance / scales[backflow], 3), 2);
      else
        element += strengths[backflow] * (2.0 * pow(distance / scales[backflow], 3) - 1.0) / pow(1.0 + pow(distance / scales[backflow], 3), 2);

    // Non-diagonal case
    if (untransformed_index == transformed_index)
      break;
  }

  return element;
}

XNum XSimulation::calculateDeterminant(XNum *matrix)
{
  double det = 1.0;
  for (int i = 0; i < N; i++)
  {
    int pivot = i;

    for (int j = i + 1; j < N; j++)
    {
      if (abs(matrix[j * N + i]) > abs(matrix[pivot * N + i]))
      {
        pivot = j;
      }
    }

    if (pivot != i)
    {

      for (int p = 0; p < N; p++)
        std::swap(matrix[i * N + p], matrix[pivot * N + p]);

      det *= -1;
    }

    if (matrix[i * N + i] == 0)
    {
      return 0;
    }

    det *= matrix[i * N + i];

    for (int j = i + 1; j < N; j++)
    {
      double factor = matrix[j * N + i] / matrix[i * N + i];
      for (int k = i + 1; k < N; k++)
      {
        matrix[j * N + k] -= factor * matrix[i * N + k];
      }
    }
  }

  return det;
}