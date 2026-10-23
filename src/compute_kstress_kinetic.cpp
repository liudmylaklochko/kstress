/* ----------------------------------------------------------------------
   compute kstress_kin : kinetic (ideal) part of sigma(q) on a list of
   wavevectors read from a file; the output rows follow the order of that file.

   sigma_id_ab(q) = -(1/V) sum_i m_i v_ia v_ib exp(-i q.r_i)
   (same sign convention as compute kstress_vir)
------------------------------------------------------------------------- */

#include "compute_kstress_kinetic.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "update.h"
#include "utils.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeKineticV3::ComputeKineticV3(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), qil(nullptr), qjm(nullptr), sgm(nullptr)
{
  if (narg != 5)
    error->all(FLERR, "Illegal compute kstress_kin command: need <qfile> <period>");

  read_qfile(arg[3]);                                    // sets nq, qil, qjm
  callBackPeriod = utils::inumeric(FLERR, arg[4], false, lmp);

  scalar_flag = vector_flag = 1;
  size_vector = 6;
  tempflag = 1;
  array_flag = 1;
  extscalar = 0;          // required by LAMMPS when scalar_flag is set
  extvector = 0;
  extarray = 0;

  size_array_rows = nq;
  size_array_cols = 6;

  memory->create(array, size_array_rows, size_array_cols, "kstress_kin:array");
  memory->create(sgm, size_array_rows, size_array_cols, "kstress_kin:sgm");
  vector = new double[6];

  if (comm->me == 0)
    utils::logmesg(lmp, "compute kstress_kin: {} wavevectors read from {}\n", nq, arg[3]);
}

/* ----------------------------------------------------------------------
   read the list of wavevectors: one line per vector, two integers il jm
------------------------------------------------------------------------- */

void ComputeKineticV3::read_qfile(const char *fname)
{
  std::vector<int> il, jm;

  FILE *fp = fopen(fname, "r");
  if (!fp) error->all(FLERR, "compute kstress_kin: cannot open q-vector file {}", fname);

  char line[512];
  int lineno = 0;
  while (fgets(line, sizeof(line), fp)) {
    lineno++;
    char *p = line;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0') continue;
    int a, b;
    if (sscanf(p, "%d %d", &a, &b) != 2)
      error->all(FLERR, "compute kstress_kin: bad line {} in q-vector file", lineno);
    il.push_back(a);
    jm.push_back(b);
  }
  fclose(fp);

  nq = (int) il.size();
  if (nq == 0) error->all(FLERR, "compute kstress_kin: q-vector file is empty");

  memory->create(qil, nq, "kstress_kin:qil");
  memory->create(qjm, nq, "kstress_kin:qjm");
  for (int k = 0; k < nq; k++) {
    qil[k] = il[k];
    qjm[k] = jm[k];
  }
}

/* ---------------------------------------------------------------------- */

ComputeKineticV3::~ComputeKineticV3()
{
  memory->destroy(sgm);
  memory->destroy(array);
  memory->destroy(qil);
  memory->destroy(qjm);
  delete[] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeKineticV3::setup()
{
  dynamic = 0;
  if (dynamic_user || group->dynamic[igroup]) dynamic = 1;
  dof_compute();
}

/* ---------------------------------------------------------------------- */

void ComputeKineticV3::dof_compute()
{
  adjust_dof_fix();
  natoms_temp = group->count(igroup);
  dof = domain->dimension * natoms_temp;
  dof -= extra_dof + fix_dof;
  if (dof > 0.0) tfactor = force->mvv2e / (dof * force->boltz);
  else tfactor = 0.0;
}

/* ---------------------------------------------------------------------- */

void ComputeKineticV3::compute_array()
{
  invoked_array = update->ntimestep;

  if ((update->ntimestep % callBackPeriod) != 0) return;

  double **x = atom->x;
  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  const double Lx = domain->xprd;
  const double Ly = domain->yprd;
  const double inv_volume = 1.0 / (Lx * Ly);
  const double k1 = M_PI / Lx;
  const double k2 = M_PI / Ly;

  for (int m = 0; m < size_array_rows; ++m)
    for (int k = 0; k < size_array_cols; ++k) {
      array[m][k] = 0.0;
      sgm[m][k] = 0.0;
    }

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    const double xi = x[i][0];
    const double yi = x[i][1];
    const double massone = rmass ? rmass[i] : mass[type[i]];

    for (int k = 0; k < nq; k++) {
      const double krj = 2.0 * (k1 * qil[k] * xi + k2 * qjm[k] * yi);
      const double A1 = -cos(krj);
      const double B1 = sin(krj);

      sgm[k][0] += massone * v[i][0] * v[i][1] * A1 * inv_volume;   // xy_re
      sgm[k][1] += massone * v[i][0] * v[i][1] * B1 * inv_volume;   // xy_im
      sgm[k][2] += massone * v[i][0] * v[i][0] * A1 * inv_volume;   // xx_re
      sgm[k][3] += massone * v[i][0] * v[i][0] * B1 * inv_volume;   // xx_im
      sgm[k][4] += massone * v[i][1] * v[i][1] * A1 * inv_volume;   // yy_re
      sgm[k][5] += massone * v[i][1] * v[i][1] * B1 * inv_volume;   // yy_im
    }
  }

  MPI_Allreduce(&sgm[0][0], &array[0][0], nq * 6, MPI_DOUBLE, MPI_SUM, world);

  for (int i = 0; i < nq; i++)
    for (int j = 0; j < 6; j++) sgm[i][j] = array[i][j];
}
