/* -*- c++ -*- ----------------------------------------------------------
   compute kstress_kin

   Kinetic (ideal) part of the wavevector-dependent stress tensor sigma(q)
   of a 2D system, on a list of wavevectors read from a file.

   Usage:
     compute ID all kstress_kin <qfile> <period>

   <qfile> : text file, one wavevector per line, two integers il jm
             (blank lines and lines starting with # are ignored).
             The physical wavevector is q = (2*pi*il/Lx, 2*pi*jm/Ly).
             Row k of the output corresponds to line k of the file, so the
             SAME file must be used for the virial part (kstress_vir)
             and when post-processing.
   <period> : same as for kstress_vir.

   Output: array of nq rows x 6 columns
             sxy_re sxy_im sxx_re sxx_im syy_re syy_im
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(kstress_kin,ComputeKineticV3);
// clang-format on
#else

#ifndef LMP_COMPUTE_KSTRESS_KINETIC_H
#define LMP_COMPUTE_KSTRESS_KINETIC_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeKineticV3 : public Compute {
 public:
  ComputeKineticV3(class LAMMPS *, int, char **);
  ~ComputeKineticV3() override;

  void init() override {}
  void setup() override;
  void compute_array() override;

  double t_target;

 private:
  void read_qfile(const char *);

  int nq;                     // number of wavevectors
  int *qil, *qjm;             // integer indices of each wavevector
  int callBackPeriod;
  double **sgm;

 protected:
  double tfactor;
  virtual void dof_compute();
};

}    // namespace LAMMPS_NS

#endif
#endif
