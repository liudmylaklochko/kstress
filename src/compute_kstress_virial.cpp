/* ----------------------------------------------------------------------
   compute kstress_vir : virial part of sigma(q) on a list of
   wavevectors read from a file; the output rows follow the order of that file.

   For each particle i the contributions of all its neighbours are summed
   into sgmtmp; when the callback moves on to another particle (or at the
   end of the timestep) finalize_atom() multiplies the buffer by the phase
   factor exp(i q.r_i) and adds it to sgm.
------------------------------------------------------------------------- */

#include "compute_kstress_virial.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "pair.h"
#include "update.h"
#include "utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeKStressTallyV3::ComputeKStressTallyV3(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), qil(nullptr), qjm(nullptr), sgm(nullptr), sgmtmp(nullptr)
{
  if (narg < 5) error->all(FLERR, "Illegal compute kstress_vir command: "
                                  "need <qfile> <period>");

  read_qfile(arg[3]);                                   // sets nq, qil, qjm
  callBackPeriod = utils::inumeric(FLERR, arg[4], false, lmp);

  scalar_flag = 1;
  vector_flag = 1;
  array_flag = 1;
  extvector = 0;
  extarray = 0;
  extscalar = 0;
  size_vector = 6;
  size_array_rows = nq;
  size_array_cols = 6;
  peratom_flag = 1;
  timeflag = 1;

  comm_reverse = size_peratom_cols = 6;
  peflag = 1;                    // we need Pair::ev_tally() to be run

  did_compute = 0;
  invoked_peratom = invoked_scalar = -1;
  nmax = -1;
  current_i = -1;

  vector = new double[size_peratom_cols];
  memory->create(array, size_array_rows, size_array_cols, "kstress_vir:array");
  memory->create(sgm, size_array_rows, size_array_cols, "kstress_vir:sgm");
  memory->create(sgmtmp, size_array_rows, size_array_cols, "kstress_vir:sgmtmp");

  if (comm->me == 0)
    utils::logmesg(lmp, "compute kstress_vir: {} wavevectors read from {}\n",
                   nq, arg[3]);
}

/* ----------------------------------------------------------------------
   read the list of wavevectors: one line per vector, two integers il jm
------------------------------------------------------------------------- */

void ComputeKStressTallyV3::read_qfile(const char *fname)
{
  std::vector<int> il, jm;

  FILE *fp = fopen(fname, "r");
  if (!fp) error->all(FLERR, "compute kstress_vir: cannot open q-vector file {}", fname);

  char line[512];
  int lineno = 0;
  while (fgets(line, sizeof(line), fp)) {
    lineno++;
    char *p = line;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0') continue;
    int a, b;
    if (sscanf(p, "%d %d", &a, &b) != 2)
      error->all(FLERR, "compute kstress_vir: bad line {} in q-vector file", lineno);
    il.push_back(a);
    jm.push_back(b);
  }
  fclose(fp);

  nq = (int) il.size();
  if (nq == 0) error->all(FLERR, "compute kstress_vir: q-vector file is empty");

  memory->create(qil, nq, "kstress_vir:qil");
  memory->create(qjm, nq, "kstress_vir:qjm");
  ilmax = jmmax = 0;
  for (int k = 0; k < nq; k++) {
    qil[k] = il[k];
    qjm[k] = jm[k];
    int a = il[k] < 0 ? -il[k] : il[k];
    int b = jm[k] < 0 ? -jm[k] : jm[k];
    if (a > ilmax) ilmax = a;
    if (b > jmmax) jmmax = b;
  }
}

/* ---------------------------------------------------------------------- */

ComputeKStressTallyV3::~ComputeKStressTallyV3()
{
  if (force && force->pair) force->pair->del_tally_callback(this);

  memory->destroy(sgm);
  memory->destroy(array);
  memory->destroy(sgmtmp);
  memory->destroy(qil);
  memory->destroy(qjm);

  delete[] vector;
}

/* ---------------------------------------------------------------------- */

void ComputeKStressTallyV3::init()
{
  if (force->pair == nullptr)
    error->all(FLERR, "Trying to use compute kstress_vir with no pair style");
  else
    force->pair->add_tally_callback(this);

  if (force->pair->single_enable == 0 || force->pair->manybody_flag)
    error->all(FLERR, "Compute kstress_vir used with incompatible pair style.");

  if ((comm->me == 0) && (force->bond || force->angle || force->dihedral ||
                          force->improper || force->kspace))
    error->warning(FLERR, "Compute kstress_vir only called from pair style");

  did_compute = -1;
  current_i = -1;
}

/* ----------------------------------------------------------------------
   apply the phase factor exp(i q.r_i) of atom i_atom to the buffer
------------------------------------------------------------------------- */

void ComputeKStressTallyV3::finalize_atom(int i_atom)
{
  double **x = atom->x;
  const double xi = x[i_atom][0];
  const double yi = x[i_atom][1];
  const double k1 = M_PI / domain->xprd;
  const double k2 = M_PI / domain->yprd;

  for (int k = 0; k < nq; k++) {
    const double kri = 2.0 * (k1 * qil[k] * xi + k2 * qjm[k] * yi);
    const double A = -cos(kri);
    const double B = sin(kri);

    for (int a = 0; a <= 2; a++) {
      sgm[k][2 * a]     += A * sgmtmp[k][2 * a]     - B * sgmtmp[k][1 + 2 * a];
      sgm[k][1 + 2 * a] += A * sgmtmp[k][1 + 2 * a] + B * sgmtmp[k][2 * a];
      sgmtmp[k][2 * a]     = 0.0;
      sgmtmp[k][1 + 2 * a] = 0.0;
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputeKStressTallyV3::pair_tally_callback(int i, int j, int nlocal, int newton_pair,
                                                double evdwl, double ecoul, double fpair,
                                                double dx, double dy, double dz)
{
  const int *const mask = atom->mask;

  if ((update->ntimestep % callBackPeriod) != 0) return;
  if (fpair == 0.0) return;

  if (did_compute != update->ntimestep) {
    did_compute = update->ntimestep;
    current_i = -1;

    if (atom->nmax > nmax) nmax = atom->nmax;

    for (int k = 0; k < nq; ++k)
      for (int l = 0; l < 6; ++l) {
        sgm[k][l] = 0.0;
        array[k][l] = 0.0;
        sgmtmp[k][l] = 0.0;
      }
  }

  // a new atom has started -> the previous one is complete
  if (i != current_i) {
    if (current_i >= 0) finalize_atom(current_i);
    current_i = i;
  }

  if (!(mask[i] & groupbit)) return;

  inv_volume = 1.0 / (domain->xprd * domain->yprd);

  const double v0 = dx * dx * fpair * inv_volume;
  const double v1 = dy * dy * fpair * inv_volume;
  const double v3 = dx * dy * fpair * inv_volume;

  const double k1 = M_PI / domain->xprd;
  const double k2 = M_PI / domain->yprd;
  const double k1x = k1 * dx;
  const double k2y = k2 * dy;
  const double small = 1.0e-8;

  // trig tables in the two directions, sized by the largest index in the list
  std::vector<double> cosx(ilmax + 1), sinx(ilmax + 1);
  std::vector<double> cosy(jmmax + 1), siny(jmmax + 1);
  for (int n = 0; n <= ilmax; n++) { cosx[n] = cos(n * k1x); sinx[n] = sin(n * k1x); }
  for (int n = 0; n <= jmmax; n++) { cosy[n] = cos(n * k2y); siny[n] = sin(n * k2y); }

  for (int k = 0; k < nq; k++) {
    const int il = qil[k];
    const int jm = qjm[k];

    const int ai = il < 0 ? -il : il;
    const int aj = jm < 0 ? -jm : jm;
    const double cos_les = cosx[ai];
    const double sin_les = il < 0 ? -sinx[ai] : sinx[ai];
    const double cos_y   = cosy[aj];
    const double sin_y   = jm < 0 ? -siny[aj] : siny[aj];

    const double krij = k1x * il + k2y * jm;

    double sss;
    if (fabs(krij) < small) sss = 1.0;
    else sss = (sin_les * cos_y + cos_les * sin_y) / krij;

    const double T_r = sss * (cos_les * cos_y - sin_les * sin_y);
    const double T_i = sss * sss * krij;

    sgmtmp[k][0] += v3 * T_r;
    sgmtmp[k][1] += v3 * T_i;
    sgmtmp[k][2] += v0 * T_r;
    sgmtmp[k][3] += v0 * T_i;
    sgmtmp[k][4] += v1 * T_r;
    sgmtmp[k][5] += v1 * T_i;
  }
}

/* ---------------------------------------------------------------------- */

void ComputeKStressTallyV3::compute_array()
{
  invoked_array = update->ntimestep;

  if (update->vflag_global != invoked_array)
    error->all(FLERR, "Virial was not tallied on needed timestep");

  // close the last atom of this timestep
  if (current_i >= 0) {
    finalize_atom(current_i);
    current_i = -1;
  }

  MPI_Allreduce(&sgm[0][0], &array[0][0], nq * 6, MPI_DOUBLE, MPI_SUM, world);

  for (int k = 0; k < nq; k++)
    for (int l = 0; l < 6; l++) sgm[k][l] = array[k][l];
}

/* ---------------------------------------------------------------------- */

double ComputeKStressTallyV3::memory_usage()
{
  double bytes = 3.0 * nq * 6 * sizeof(double) + 2.0 * nq * sizeof(int);
  return bytes;
}
