/* -*- c++ -*- ----------------------------------------------------------
   compute kstress_vir

   Virial (pair) part of the wavevector-dependent stress tensor sigma(q)
   of a 2D system, on a list of wavevectors read from a file.

   Usage:
     compute ID all kstress_vir <qfile> <period>

   <qfile> : text file, one wavevector per line, two integers il jm
             (blank lines and lines starting with # are ignored)
                 # il  jm
                   4   0
                   0   4
                  -4   4
             The physical wavevector is q = (2*pi*il/Lx, 2*pi*jm/Ly).
             Row k of the output array corresponds to line k of the file,
             so the SAME file must be given to compute kstress_kin and to the
             correlator (correlation_kstress.cpp).
   <period> : sigma(q) is computed on timesteps that are multiples of it.

   Output: array of nq rows x 6 columns
             sxy_re sxy_im sxx_re sxx_im syy_re syy_im

   Restrictions: 2D, orthogonal box, newton_pair on (LAMMPS default),
   pair forces only (no many-body, bond, angle, dihedral or kspace terms),
   pair styles that call the tally interface (not GPU/KOKKOS/INTEL).
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(kstress_vir,ComputeKStressTallyV3);
// clang-format on
#else

#ifndef LMP_COMPUTE_KSTRESS_VIRIAL_H
#define LMP_COMPUTE_KSTRESS_VIRIAL_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeKStressTallyV3 : public Compute {
 public:
  ComputeKStressTallyV3(class LAMMPS *, int, char **);
  ~ComputeKStressTallyV3() override;

  void init() override;
  void pair_tally_callback(int, int, int, int, double, double, double,
                           double, double, double) override;
  void compute_array() override;
  double memory_usage() override;

 private:
  void finalize_atom(int);            // apply exp(i q.r_i) to the buffer
  void read_qfile(const char *);      // read the list of wavevectors

  int nq;                             // number of wavevectors
  int *qil, *qjm;                     // integer indices of each wavevector
  int ilmax, jmmax;                   // largest |il|, |jm| (for trig tables)

  int callBackPeriod;
  int did_compute;
  int current_i;                      // atom currently being accumulated
  int nmax;
  double inv_volume;

  double **sgm;                       // accumulated sigma(q)
  double **sgmtmp;                    // buffer for the current atom
};

}    // namespace LAMMPS_NS

#endif
#endif

