/*
 * correlation_kstress.cpp : stress correlation functions C(q,t) from the
 * output of the LAMMPS computes kstress_vir and kstress_kin
 * (multiple-tau correlator). Independent of LAMMPS.
 *
 * Compile:
 *   g++ -O2 -fopenmp -o cort correlation_kstress.cpp -lm
 *
 * Usage:
 *   ./cort <ms> qlist=<file> <keyword=file> [keyword=file ...]     (kstress_vir/kstress_kin)
 *   ./cort <ms> qn=<step> <keyword=file> [keyword=file ...]        (full-grid data, earlier code)
 *
 * Lag time = (timestep difference) * dt; dt= sets the MD timestep
 * (default 0.005). The projections assume a square box, Lx = Ly.
 *
 * ---------------------------------------------------------------------------
 * TWO MODES, depending on which compute produced the data
 * ---------------------------------------------------------------------------
 *
 * (a) GRID MODE  -- data from the earlier full-grid computes
 *
 *     The compute stores a full grid, row = jl + nl*(qmax+1), and qmax is
 *     auto-detected here from the number of rows, li2 = (qmax+1)(2*qmax+1).
 *     qn is the compute's dq argument, so that the real integer wavevectors
 *     are multiples of qn. Nothing else is needed:
 *
 *       ./cort 50 qn=4 kin=kstress.kinetic.dat exc=kstress.excess.dat
 *
 * (b) Q-LIST MODE -- data from compute kstress_vir + kstress_kin (this package)
 *
 *     Those computes store one row per line of the q-vector file they were
 *     given, in that file's order. Pass THE SAME FILE here with qlist=:
 *
 *       ./cort 50 qlist=qlist.dat kin=kin.dat exc=exc.dat
 *
 *     The list may be arbitrary: the vectors need not lie on a regular grid,
 *     need not share a common spacing, and jm may be negative. The number of
 *     rows in the data file must equal the number of lines in the list (this
 *     is checked). In this mode qn is NOT needed (the list holds the real
 *     integer wavevectors) and qlim defaults to the largest |q| in the list.
 *
 *     Format of the q-vector file (as written by make_qlist.py):
 *         # il  jm
 *           -120   0
 *              0   4
 *            137  64
 *     Blank lines and lines starting with # are ignored. il and jm are the
 *     REAL integer wavevectors, q = (2*pi*il/Lx, 2*pi*jm/Ly).
 *
 *     Note: shells are averaged over the vectors present in the list. If a
 *     shell is only partly represented (e.g. (4,0) listed but not (0,4)), the
 *     average uses fewer vectors and is correspondingly noisier.
 *
 * ---------------------------------------------------------------------------
 * Keywords (activate only what you have data for)
 * ---------------------------------------------------------------------------
 *   kin=<file>    kinetic stress   (compute kstress_kin; row index + 6 cols)
 *   exc=<file>    virial stress    (compute kstress_vir; row index + 6 cols)
 *   cqch=<file>   concentration + kinetic-energy fluct (compute cqch; 5 cols/row)
 *   psiq=<file>   concentration + size fluctuation     (compute psiq; 5 cols/row)
 *   qlist=<file>  explicit q-vector list; REQUIRED for v3 data, and must be
 *                 the same file that was given to the compute
 *   qlim=<q>      restrict the analysis to |q| <= qlim (default: everything)
 *   out=<prefix>  output filename prefix (default "newcorr")
 *   parts         (flag) also correlate the ideal & virial stress parts
 *                 separately: sigma = sigma^id (kin) + sigma^vir (exc)
 *   shells=<q2,q2,...>  analyse only these |q|^2 shells (e.g. shells=16,28800)
 *   absC          (flag) accumulate |C(t)| = sqrt(Re^2+Im^2) of s22s22 for EACH
 *                 q-vector of the shell before the shell average (the Im column
 *                 of s22s22 is then 0). Use on raw data with a centre-of-mass
 *                 drift: for a constant drift |C| per vector is drift-free.
 *                 Other channels are left unchanged. Use out= to keep both.
 *   muF           (flag) also compute the affine modulus mu_F(q), RAW (no V/T):
 *                 mu_F = V/T * [ <|sD|^2> - |<sD>|^2 ],  sD = 1/2(s11 - s22)
 *                 written to muF_Qmax<qlim>.dat (one value per shell)
 *
 * Channels computed (only for the data provided):
 *   kin+exc          -> s12s12, s11s11, s22s22, s11s22          (full stress)
 *   kin+exc + parts  -> + id12id12, id11id11, id22id22, id11id22   (ideal part)
 *                       + vir12vir12, vir11vir11, vir22vir22, vir11vir22 (virial)
 *   cqch             -> cc, ee            (+ s11c, s22c, s11e, s22e with stress)
 *   psiq             -> kk                (+ s11k, s22k with stress)
 *
 * Convention: all correlations are <A*(q,0) B(q,t)> (no V/T factor applied).
 *
 * Output: <out>.Qmax<qlim>.q<q2>.dat, one file per |q|^2 shell
 *         (<out> is set by out=, default "newcorr"). Use different prefixes
 *         when analysing several sampling stages of the same run, e.g.
 *           out=corr_20tau  for the stage sampled every 20 tau_LJ
 *           out=corr_02tau  for the stage sampled every 0.2 tau_LJ
 *         otherwise the second run overwrites the first.
 *   col 1: time; then (re, im) pairs per active channel, in header order.
 *
 * Examples:
 *   ./cort 50 qn=4 kin=kstress.kinetic.dat exc=kstress.excess.dat parts
 *   ./cort 50 qn=4 kin=kin.dat exc=exc.dat parts muF
 *   ./cort 50 qn=1 kin=kin.dat exc=exc.dat qlim=30
 *   ./cort 50 qn=4 kin=kin.dat exc=exc.dat cqch=cqch.dat psiq=psi.dat parts
 *   ./cort 50 qlist=qlist.dat kin=kin.dat exc=exc.dat parts        (v3 data)
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <set>
#include <algorithm>
#include <ctime>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

static int QN = 1;
static double DT = 0.005;           /* MD timestep (dt= keyword): lag time = steps * DT */

/* ---------- memory ---------- */
double ***alloc3(int x, int y, int z) {
    double *p = (double *) calloc((size_t)x * y * z, sizeof(double));
    double ***a = (double ***) malloc(x * sizeof(double **));
    for (int i = 0; i < x; i++) {
        a[i] = (double **) malloc(y * sizeof(double *));
        for (int j = 0; j < y; j++) a[i][j] = &p[(size_t)z * j + (size_t)y * z * i];
    }
    return a;
}
double *alloc1(int x) { return (double *) calloc(x, sizeof(double)); }
double **alloc2(int x, int y) {
    double **p = (double **) malloc(sizeof(double *) * (x + 5));
    for (int i = 0; i < x + 5; i++) p[i] = (double *) calloc(y, sizeof(double));
    return p;
}

/* ---------- multiple-tau correlator ---------- */

void zeros(int klast, int nlast, int ms, int Nf,
           double **ct, double **ct_im, double **ct_re_im, double **ct_im_re)
{
    if (Nf % 2 != 0) Nf--;
    for (int n = 0; n <= 2 * ms; n++) {
        ct[0][n] = ct_im[0][n] = ct_re_im[0][n] = ct_im_re[0][n] = 0;
    }
    for (int k = 1; k < klast; k++)
        for (int n = ms + 1; n <= 2 * ms; n++) {
            ct[k][n] = ct_im[k][n] = ct_re_im[k][n] = ct_im_re[k][n] = 0;
        }
    for (int n = ms + 1; n < nlast; n++) {
        ct[klast][n] = ct_im[klast][n] = ct_re_im[klast][n] = ct_im_re[klast][n] = 0;
    }
}

/* generic cross-correlation of complex signals A and B; A==B gives autocorrelation */
void cross_correlation(int klast, int nlast, int ms, int Nf, int maxrowline,
                       const double *A_re, const double *A_im,
                       const double *B_re, const double *B_im,
                       double **ct, double **ct_im, double **ct_re_im, double **ct_im_re)
{
    double *ar = alloc1(maxrowline), *ai = alloc1(maxrowline);
    double *br = alloc1(maxrowline), *bi = alloc1(maxrowline);

    if (Nf % 2 != 0) Nf--;

    for (int n = 0; n <= 2 * ms; n++) {
        for (int i = 0; i < Nf - n; i++) {
            ct[0][n]       += A_re[i] * B_re[n + i];
            ct_im[0][n]    += A_im[i] * B_im[n + i];
            ct_re_im[0][n] += A_re[i] * B_im[n + i];
            ct_im_re[0][n] += A_im[i] * B_re[n + i];
        }
        ct[0][n] /= (Nf - n); ct_im[0][n] /= (Nf - n);
        ct_re_im[0][n] /= (Nf - n); ct_im_re[0][n] /= (Nf - n);
    }

    int Nk = Nf / 2;
    for (int i = 0; i < Nk; i++) {
        ar[i] = (A_re[2*i] + A_re[2*i+1]) / 2;  ai[i] = (A_im[2*i] + A_im[2*i+1]) / 2;
        br[i] = (B_re[2*i] + B_re[2*i+1]) / 2;  bi[i] = (B_im[2*i] + B_im[2*i+1]) / 2;
    }
    for (int n = ms + 1; n <= 2 * ms; n++) {
        for (int i = 0; i < Nk - n; i++) {
            ct[1][n]       += ar[i] * br[i + n];
            ct_im[1][n]    += ai[i] * bi[i + n];
            ct_re_im[1][n] += ar[i] * bi[i + n];
            ct_im_re[1][n] += ai[i] * br[i + n];
        }
        ct[1][n] /= (Nk - n); ct_im[1][n] /= (Nk - n);
        ct_re_im[1][n] /= (Nk - n); ct_im_re[1][n] /= (Nk - n);
    }

    int k = 2;
    while ((ms * pow(2, k) < Nf) && (k < klast)) {
        Nk /= 2;
        for (int i = 0; i < Nk; i++) {
            ar[i] = (ar[2*i] + ar[2*i+1]) / 2;  ai[i] = (ai[2*i] + ai[2*i+1]) / 2;
            br[i] = (br[2*i] + br[2*i+1]) / 2;  bi[i] = (bi[2*i] + bi[2*i+1]) / 2;
        }
        for (int n = ms + 1; n <= 2 * ms; n++) {
            for (int i = 0; i < Nk - n; i++) {
                ct[k][n]       += ar[i] * br[i + n];
                ct_im[k][n]    += ai[i] * bi[i + n];
                ct_re_im[k][n] += ar[i] * bi[i + n];
                ct_im_re[k][n] += ai[i] * br[i + n];
            }
            ct[k][n] /= (Nk - n); ct_im[k][n] /= (Nk - n);
            ct_re_im[k][n] /= (Nk - n); ct_im_re[k][n] /= (Nk - n);
        }
        k++;
    }

    k = klast; Nk /= 2;
    for (int i = 0; i < Nk; i++) {
        ar[i] = (ar[2*i] + ar[2*i+1]) / 2;  ai[i] = (ai[2*i] + ai[2*i+1]) / 2;
        br[i] = (br[2*i] + br[2*i+1]) / 2;  bi[i] = (bi[2*i] + bi[2*i+1]) / 2;
    }
    for (int n = ms + 1; n < nlast; n++) {
        for (int i = 0; i < Nk - n; i++) {
            ct[k][n]       += ar[i] * br[i + n];
            ct_im[k][n]    += ai[i] * bi[i + n];
            ct_re_im[k][n] += ar[i] * bi[i + n];
            ct_im_re[k][n] += ai[i] * br[i + n];
        }
        ct[k][n] /= (Nk - n); ct_im[k][n] /= (Nk - n);
        ct_re_im[k][n] /= (Nk - n); ct_im_re[k][n] /= (Nk - n);
    }

    free(ar); free(ai); free(br); free(bi);
}

/* ---------- q-vector indexing ---------- */

/* ---------- explicit q-vector list (compute kstress_vir / kstress_kin) ----
   v3 writes one row per line of its qlist file, in that file's order, so the
   grid formula below no longer applies. If qlist=<file> is given, the same
   file must be used here: row k of the data = line k of the list.
   Without qlist=, everything behaves exactly as before (v2 grid).        */
static vector<int> QLIST_IL, QLIST_JM;      // empty => v2 grid mode
static bool QLIM_USER = false;               // true if qlim= was given on the command line

void read_qlist(const string &fname) {
    FILE *f = fopen(fname.c_str(), "r");
    if (!f) { fprintf(stderr, "ERROR: cannot open qlist file %s\n", fname.c_str()); exit(1); }
    char line[512];
    int lineno = 0;
    while (fgets(line, sizeof(line), f)) {
        lineno++;
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0') continue;
        int a, b;
        if (sscanf(p, "%d %d", &a, &b) != 2) {
            fprintf(stderr, "ERROR: bad line %d in qlist file %s\n", lineno, fname.c_str());
            exit(1);
        }
        QLIST_IL.push_back(a);
        QLIST_JM.push_back(b);
    }
    fclose(f);
    if (QLIST_IL.empty()) { fprintf(stderr, "ERROR: qlist file %s is empty\n", fname.c_str()); exit(1); }
    fprintf(stderr, "qlist: %zu wavevectors read from %s\n", QLIST_IL.size(), fname.c_str());
}

/* Row layout written by the earlier full-grid computes (and cqch/psiq):
   row = jl + nl*(qmax+1), with nl = [0..2*qmax], jl = [0..qmax],
   holding the real integer wavevector ((nl-qmax)*qn, jl*qn).
   Hence for a real (qx,qy) (multiples of qn):
   nl = qx/qn + qmax, jl = qy/qn.
   With an explicit q-list (v3), the row is instead the position of (qx,qy)
   in that list; -1 is returned if the vector is not in it.              */
int find_index(int qx, int qy, int qmax, int qn) {
    if (!QLIST_IL.empty()) {                       /* ---- v3 list mode ---- */
        for (size_t k = 0; k < QLIST_IL.size(); k++)
            if (QLIST_IL[k] == qx && QLIST_JM[k] == qy) return (int)k;
        return -1;                                 /* not in the list */
    }
    if (qx % qn != 0 || qy % qn != 0) {
        fprintf(stderr, "ERROR: (%d,%d) not on the qn=%d grid\n", qx, qy, qn);
        exit(1);
    }
    int nl = qx / qn + qmax;
    int jl = qy / qn;
    if (nl < 0 || nl > 2 * qmax || jl < 0 || jl > qmax) {
        fprintf(stderr, "ERROR: (%d,%d) outside stored grid (qmax=%d qn=%d)\n",
                qx, qy, qmax, qn);
        exit(1);
    }
    return jl + nl * (qmax + 1);
}

/* Build the |q|^2 shell list over real wavevectors
   qx,qy = multiples of qn within [-qlim, +qlim] (qy >= 0 half-plane).
   qlim is the analysis range and is independent of the file's qmax. */
void generate_qqq(int qlim, int qn, vector<int> &qqq) {
    set<int> s;
    if (!QLIST_IL.empty()) {          /* shells present in the explicit list */
        /* qlim here is only a filter if the user gave it explicitly; the
           default (axis reach) must not cut off the corner shells */
        for (size_t k = 0; k < QLIST_IL.size(); k++) {
            int q2 = QLIST_IL[k]*QLIST_IL[k] + QLIST_JM[k]*QLIST_JM[k];
            if (q2 > 0 && (!QLIM_USER || q2 <= qlim*qlim)) s.insert(q2);
        }
    } else {                          /* full v2 grid */
        for (int qx = -qlim; qx <= qlim; qx += qn)
            for (int qy = qn; qy <= qlim; qy += qn) s.insert(qx * qx + qy * qy);
        for (int qx = qn; qx <= qlim; qx += qn) s.insert(qx * qx);
    }
    qqq.assign(s.begin(), s.end());
    sort(qqq.begin(), qqq.end());
}

/* ---------- file reading (fix ave/time mode vector format) ---------- */

/* ---------- progress bar ---------- */
static void draw_progress(int done, int total, double t_start) {
    const int width = 40;
    double frac = total > 0 ? (double)done / total : 1.0;
    int filled = (int)(frac * width);
    double elapsed = (double)(clock()) / CLOCKS_PER_SEC; (void)elapsed;
    double wall = difftime(time(NULL), (time_t)t_start);
    double eta = (done > 0) ? wall * (total - done) / done : 0.0;
    fprintf(stderr, "\r[");
    for (int i = 0; i < width; i++) fputc(i < filled ? '#' : '-', stderr);
    fprintf(stderr, "] %3d%%  %d/%d shells  elapsed %.0fs  ETA %.0fs   ",
            (int)(frac * 100), done, total, wall, eta);
    fflush(stderr);
}

/* count frames in a "fix ave/time ... mode vector" file: lines with exactly
   two integer tokens (timestep, nrows) */
static long long count_frames(const string &fname) {
    FILE *f = fopen(fname.c_str(), "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname.c_str()); exit(1); }
    char line[4096]; long long n = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#') continue;
        long long a; int b; char extra;
        if (sscanf(line, "%lld %d %c", &a, &b, &extra) == 2) n++;
    }
    fclose(f);
    return n;
}

long long read_datafile(const string &fname, double ***arr, int ncol, int maxrowline,
                        long long &li2_out, int *timevect, bool fill_timevect)
{
    FILE *f = fopen(fname.c_str(), "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", fname.c_str()); exit(1); }

    fseek(f, 0, SEEK_END);
    long long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    time_t rd_start = time(NULL);
    fprintf(stderr, "Loading %s ...\n", fname.c_str());

    char line[2048];
    long position = 0;
    long long li1 = 0, li2 = 0;
    while (fgets(line, sizeof(line), f) != NULL) {
        if (line[0] != '#') {
            char *p = line;
            li1 = strtoll(p, &p, 10);
            li2 = strtoll(p, &p, 10);
            break;
        }
        position = ftell(f);
    }
    (void)li1;
    li2_out = li2;
    fseek(f, position, SEEK_SET);

    long long t = 0;
    int time_val, nq;
    while (!feof(f)) {
        if (fscanf(f, "%d %d", &time_val, &nq) != 2) break;
        if (fill_timevect) timevect[t] = time_val;
        for (long long i = 0; i < li2; i++) {
            int got = 0;
            for (int c = 0; c < ncol; c++)
                got += fscanf(f, "%lf", &arr[c][i][t]);
            if (got != ncol) { fclose(f); return t; }
        }
        t++;
        if (t % 200 == 0) {
            long long pos = ftell(f);
            double frac = fsize > 0 ? (double)pos / fsize : 1.0;
            int W = 40, fill = (int)(frac * W);
            double wall = difftime(time(NULL), rd_start);
            fprintf(stderr, "\r  [");
            for (int i = 0; i < W; i++) fputc(i < fill ? '=' : '-', stderr);
            fprintf(stderr, "] %3d%%  %lld frames  %.0fs   ", (int)(frac*100), t, wall);
            fflush(stderr);
        }
        if (t >= maxrowline) { fprintf(stderr, "WARNING: t reached maxrowline=%d\n", maxrowline); break; }
    }
    fclose(f);
    fprintf(stderr, "\r  [========================================] 100%%  %lld frames loaded   \n", t);
    return t;
}

/* ---------- channel bookkeeping ---------- */

struct Channel {
    string name;                    // e.g. "s11s22"
    const double *A_re, *A_im;      // set per q-vector
    const double *B_re, *B_im;
    double **ct, **ct_im, **ct_re_im, **ct_im_re;   // scratch
    double **acc_re, **acc_im;                       // shell accumulators
};


/* ========================= MAIN ========================= */

int main(int argc, char **argv) {

    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <ms> qlist=<file> keyword=file [keyword=file ...]\n"
            "   or: %s <ms> qn=<step> keyword=file [keyword=file ...]   (full-grid data)\n"
            "  qn=<step>   real wavevector spacing = the compute's dq argument\n"
            "              (qn=1 for dq=1 runs, qn=4 for dq=4 runs). REQUIRED.\n"
            "  qmax is auto-detected from the file's row count.\n"
            "  keywords: kin= exc= cqch= psiq= parts qlim=<real q limit>\n"
            "            qlist=<file>  explicit q-vector list (kstress_vir/kstress_kin);\n"
            "                          MUST be the same file used in the run\n"
            "            muF\n"
            "            dt=<x>    MD timestep for the lag time (default 0.005)\n"
            "            shells=16,28800  analyse only these |q|^2 shells\n"
            "            absC      = |C(t)| of s22s22 per q-vector before the shell average\n"
            "                        (drift test; combine with out=abscorr)\n"
            "  parts = also output ideal (kin) & virial (exc) stress correlations\n"
            "  muF       = also output RAW mu_F(q) variances (no V/T; apply it downstream):\n"
            "              muF_rot = Var(1/2(s11-s22))  in the q-rotated frame\n"
            "              muF0    = Var(sxy)           0 deg shear\n"
            "              muF45   = Var(1/2(sxx-syy))  45 deg shear\n"
            "              muF_ori = (muF0+muF45)/2     <- compare with reference table\n"
            "\n"
            "Examples:\n"
            "  # stress only, dq=4 run, q in [-120,120]\n"
            "  %s 50 qn=4 kin=kstress.kinetic.qmax120.dat exc=kstress.virial.qmax120.dat\n"
            "\n"
            "  # add mu_F(q)\n"
            "  %s 50 qn=4 kin=kin.dat exc=exc.dat muF\n"
            "\n"
            "  # mu_F + ideal/virial decomposition (what av_corr.py expects)\n"
            "  %s 50 qn=4 kin=kin.dat exc=exc.dat parts muF\n"
            "\n"
            "  # everything, restricted to q <= 60\n"
            "  %s 50 qn=4 kin=kin.dat exc=exc.dat cqch=cqch.dat psiq=psi.dat parts muF qlim=60\n"
            "\n"
            "  # dq=1 run (fine grid for the small-q plateau)\n"
            "  %s 50 qn=1 kin=kin.dat exc=exc.dat muF\n",
            argv[0], argv[0], argv[0], argv[0], argv[0], argv[0], argv[0]);
        return 1;
    }

    int ms = atoi(argv[1]);

    string f_kin, f_exc, f_cqch, f_psiq;
    int qlim = -1;                    /* analysis range in real q; -1 = default (full grid) */
    string OUTPREF = "newcorr";       /* output filename prefix (out= keyword) */
    bool want_parts = false;          /* also correlate ideal & virial parts separately */
    bool want_muF = false;             /* affine modulus mu_F(q), RAW (no V/T here) */
    bool want_absC = false;            /* |C| per q-vector for s22s22 (drift test) */
    set<int> only_shells;              /* shells=16,28800 : analyse only these |q|^2 */
    QN = 0;                           /* real q spacing; must be given via qn= */
    for (int a = 2; a < argc; a++) {
        string s = argv[a];
        if      (s.rfind("kin=",  0) == 0) f_kin  = s.substr(4);
        else if (s.rfind("exc=",  0) == 0) f_exc  = s.substr(4);
        else if (s.rfind("cqch=", 0) == 0) f_cqch = s.substr(5);
        else if (s.rfind("psiq=", 0) == 0) f_psiq = s.substr(5);
        else if (s.rfind("qlim=", 0) == 0) { qlim = atoi(s.substr(5).c_str()); QLIM_USER = true; }
        else if (s.rfind("qn=",   0) == 0) QN     = atoi(s.substr(3).c_str());
        else if (s.rfind("qlist=", 0) == 0) read_qlist(s.substr(6));
        else if (s.rfind("out=",   0) == 0) OUTPREF = s.substr(4);
        else if (s.rfind("dt=",    0) == 0) DT = atof(s.substr(3).c_str());
        else if (s == "muF")               want_muF = true;
        else if (s == "parts")             want_parts = true;
        else if (s == "absC")              want_absC = true;
        else if (s.rfind("shells=", 0) == 0) {
            string v = s.substr(7); size_t p0 = 0;
            while (p0 < v.size()) {
                size_t p1 = v.find(',', p0); if (p1 == string::npos) p1 = v.size();
                only_shells.insert(atoi(v.substr(p0, p1 - p0).c_str())); p0 = p1 + 1;
            }
        }
        else { fprintf(stderr, "Unknown keyword: %s\n", s.c_str()); return 1; }
    }
    if (!QLIST_IL.empty()) {
        /* q-list mode: the integers in the file ARE the real wavevectors, so
           qn has no role. Accept it silently if given, default it otherwise. */
        if (QN <= 0) QN = 1;
    } else if (QN <= 0) {
        fprintf(stderr, "ERROR: you must give qn=<step> (the compute's dq; qn=1 or qn=4)\n"
                        "       (not needed when qlist=<file> is given)\n");
        return 1;
    }
    /* qmax is derived from the file row count after loading (see below). */
    int qmax = -1;

    bool have_stress = !f_kin.empty() && !f_exc.empty();
    bool have_cqch   = !f_cqch.empty();
    bool have_psiq   = !f_psiq.empty();
    if (!have_stress && (!f_kin.empty() || !f_exc.empty())) {
        fprintf(stderr, "Stress needs BOTH kin= and exc=\n"); return 1;
    }
    if (!have_stress && !have_cqch && !have_psiq) {
        fprintf(stderr, "Nothing to do: give kin=+exc= and/or cqch= and/or psiq=\n"); return 1;
    }

    printf("ms=%d, qn=%d (qmax auto-detected from file below)\n", ms, QN);
    bool have_parts = have_stress && want_parts;
    bool have_muF   = have_stress && want_muF;
    if (want_muF && !have_stress) {
        fprintf(stderr, "ERROR: muF needs stress data (kin= and exc=)\n");
        return 1;
    }
    printf("Stress: %s | parts(id+vir): %s | cqch: %s | psiq: %s | absC(s22s22): %s\n",
           have_stress ? "ON" : "off", have_parts ? "ON" : "off",
           have_cqch ? "ON" : "off", have_psiq ? "ON" : "off", want_absC ? "ON" : "off");

    /* frame capacity: count the frames in every input file and take the largest,
       so nothing is ever silently truncated. (Old versions hard-coded 10000.) */
    long long nf_max = 0;
    for (const string &fn : {f_kin, f_exc, f_cqch, f_psiq})
        if (!fn.empty()) { long long n = count_frames(fn); if (n > nf_max) nf_max = n; }
    if (nf_max <= 0) { fprintf(stderr, "ERROR: no frames found in the input files\n"); return 1; }
    const int maxrowline = (int) nf_max + 1;
    printf("Frames in input: %lld (arrays sized accordingly)\n", nf_max);
    const int ncol = 7;

    int *timevect = (int *) calloc(maxrowline, sizeof(int));
    long long li2 = 0, t = 0;

    double ***kin = NULL, ***exc = NULL, ***tot = NULL;

    if (have_stress) {
        // need li2 first: peek header of kin file
        kin = NULL;
        {   // read kin
            FILE *f = fopen(f_kin.c_str(), "r");
            if (!f) { fprintf(stderr, "Cannot open %s\n", f_kin.c_str()); return 1; }
            char line[2048]; long long a=0,b=0;
            while (fgets(line, sizeof(line), f)) {
                if (line[0] != '#') { char *p=line; a=strtoll(p,&p,10); b=strtoll(p,&p,10); break; }
            }
            fclose(f); li2 = b; (void)a;
        }
        kin = alloc3(ncol, li2, maxrowline);
        exc = alloc3(ncol, li2, maxrowline);
        tot = alloc3(ncol, li2, maxrowline);
        long long dummy;
        t = read_datafile(f_kin, kin, ncol, maxrowline, dummy, timevect, true);
        long long t2 = read_datafile(f_exc, exc, ncol, maxrowline, dummy, timevect, false);
        printf("kin: %lld frames, exc: %lld frames, q-rows=%lld\n", t, t2, li2);
        if (t != t2) { fprintf(stderr, "ERROR: kin/exc frame mismatch\n"); return 1; }
        for (long long i = 0; i < t; i++)
            for (long long j = 0; j < li2; j++)
                for (int c = 0; c < 6; c++)
                    tot[c][j][i] = kin[c + 1][j][i] + exc[c + 1][j][i];
    }

    auto read_aux = [&](const string &fname, double ***&arr, const char *tag) {
        long long li2x = 0;
        {   FILE *f = fopen(fname.c_str(), "r");
            if (!f) { fprintf(stderr, "Cannot open %s\n", fname.c_str()); exit(1); }
            char line[2048]; long long a=0,b=0;
            while (fgets(line, sizeof(line), f)) {
                if (line[0] != '#') { char *p=line; a=strtoll(p,&p,10); b=strtoll(p,&p,10); break; }
            }
            fclose(f); li2x = b; (void)a;
        }
        arr = alloc3(5, li2x, maxrowline);   // 5 cols: rowidx + 4 data
        long long dummy;
        long long tx = read_datafile(fname, arr, 5, maxrowline, dummy, timevect, t == 0);
        printf("%s: %lld frames, q-rows=%lld\n", tag, tx, li2x);
        if (t > 0 && (tx != t || li2x != li2)) {
            fprintf(stderr, "ERROR: %s frames/q-rows do not match other files\n", tag);
            exit(1);
        }
        if (t == 0) { t = tx; li2 = li2x; }
        /* convention fix: negate sin-columns (cols 2 and 4) */
        for (long long i = 0; i < t; i++)
            for (long long j = 0; j < li2x; j++) {
                arr[2][j][i] = -arr[2][j][i];
                arr[4][j][i] = -arr[4][j][i];
            }
    };
    double ***qch = NULL, ***psq = NULL;
    if (have_cqch) read_aux(f_cqch, qch, "cqch");
    if (have_psiq) read_aux(f_psiq, psq, "psiq");

    /* derive qmax from the file row count: li2 = (qmax+1)(2*qmax+1)
       => qmax = (-3 + sqrt(9 - 8(1 - li2))) / 4                       */
    if (!QLIST_IL.empty()) {
        /* explicit q-list (compute v3): the file has one row per list entry */
        if ((long long)QLIST_IL.size() != li2) {
            fprintf(stderr,
                "ERROR: the data file has %lld q-rows but the qlist file has "
                "%zu wavevectors. They must be the same file used in the run.\n",
                li2, QLIST_IL.size());
            return 1;
        }
        qmax = 0;                       /* unused in list mode */
        printf("q-list mode: %lld wavevectors, taken from the qlist file.\n", li2);
    } else {
        double disc = 9.0 - 8.0 * (1.0 - (double)li2);
        double qd   = (-3.0 + sqrt(disc)) / 4.0;
        qmax = (int)llround(qd);
        long long expect_rows = (long long)(qmax + 1) * (2 * qmax + 1);
        if (qmax < 1 || expect_rows != li2) {
            fprintf(stderr,
                "ERROR: file has %lld q-rows, which is not a valid "
                "(qmax+1)(2*qmax+1) grid. File is not a kstress-layout file, "
                "or is truncated.\n", li2);
            return 1;
        }
        printf("Auto-detected stored_qmax=%d from %lld rows; qn=%d "
               "-> real qmax=%d.\n", qmax, li2, QN, qmax * QN);
    }

    /* analysis range: default = full stored grid; otherwise validate the given qlim */
    if (qlim < 0) {
        if (!QLIST_IL.empty()) {
            /* largest index component in the list (the axis reach), so that a
               full-grid list is labelled exactly as grid mode would: the corner
               (+-120,120) gives Qmax120, not Qmax170. Every shell in the list
               is analysed regardless; this value is only the label. */
            int amax = 0;
            for (size_t k = 0; k < QLIST_IL.size(); k++) {
                int a = std::abs(QLIST_IL[k]), b = std::abs(QLIST_JM[k]);
                if (a > amax) amax = a;
                if (b > amax) amax = b;
            }
            qlim = amax;
        } else qlim = qmax * QN;
    }
    if (QLIST_IL.empty() && qlim % QN != 0) {
        fprintf(stderr, "ERROR: qlim=%d is not a multiple of qn=%d\n", qlim, QN);
        return 1;
    }
    if (QLIST_IL.empty() && qlim > qmax * QN) {
        fprintf(stderr, "ERROR: qlim=%d exceeds what the file stores (max %d)\n",
                qlim, qmax * QN);
        return 1;
    }

    int Nf = (int) t;

    /* shift time origin to zero: if the first recorded timestep is not 0
       (e.g. output started at step 100000), subtract it from all entries
       so the correlation time axis starts at 0 */
    if (Nf > 0 && timevect[0] != 0) {
        int t0 = timevect[0];
        printf("First timestep is %d, shifting time axis to start at 0\n", t0);
        for (int i = 0; i < Nf; i++) timevect[i] -= t0;
    }

    int klast = (int)(log((double)(Nf / (2 * ms))) / log(2.0));
    int nlast = (int)(Nf / pow(2, klast));
    printf("Nf=%d klast=%d nlast=%d\n", Nf, klast, nlast);

    vector<int> qqq;
    generate_qqq(qlim, QN, qqq);
    if (!only_shells.empty()) {          /* shells= : keep only the requested |q|^2 */
        vector<int> keep;
        for (int q2 : qqq) if (only_shells.count(q2)) keep.push_back(q2);
        qqq = keep;
        printf("shells= : analysing %zu shell(s) only\n", qqq.size());
    }
    int nqqq = qqq.size();
    printf("File grid: qmax=%d qn=%d -> real q in [-%d,%d] step %d (%lld rows)\n",
           qmax, QN, qmax * QN, qmax * QN, QN, (long long)li2);
    printf("Analysis range: q in [-%d,%d] step %d\n", qlim, qlim, QN);
    printf("Unique |q|^2 shells: %d (%d..%d)\n", nqqq, qqq.front(), qqq.back());
    double *dqcnt = (double *) calloc(nqqq, sizeof(double));
    /* mu_F(q): one scalar per shell (sum over the shell's q-vectors of the
       connected variance of sigma_D = 1/2 (sigma11 - sigma22)); averaged by
       dqcnt and scaled by V/T at output. Shared across threads, but each shell
       qv is handled by exactly one thread, so plain writes are race-free. */
    double *muF_acc   = (double *) calloc(nqqq, sizeof(double));  /* rotated  : 1/2(s11-s22) */
    double *muF0_acc  = (double *) calloc(nqqq, sizeof(double));  /* 0 deg: sxy          */
    double *muF45_acc = (double *) calloc(nqqq, sizeof(double));  /* 45deg: 1/2(sxx-syy) */
    double *shell_nq = (double *) calloc(nqqq, sizeof(double));  /* q-vectors per shell */

    /* ---------- per-thread workspace ---------- */
    struct Workspace {
        double *s11r, *s11i, *s22r, *s22i, *s12r, *s12i;
        double *ccr, *cci, *eer, *eei, *kkr, *kki;
        double *mfr, *mfi;                 /* scratch for mu_F variances */
        double *i11r,*i11i,*i22r,*i22i,*i12r,*i12i;   /* ideal (kinetic) part */
        double *v11r,*v11i,*v22r,*v22i,*v12r,*v12i;   /* virial (excess) part */
        vector<Channel> ch;
    };

    auto build_workspace = [&](Workspace &w) {
        w.s11r = alloc1(maxrowline); w.s11i = alloc1(maxrowline);
        w.s22r = alloc1(maxrowline); w.s22i = alloc1(maxrowline);
        w.s12r = alloc1(maxrowline); w.s12i = alloc1(maxrowline);
        w.ccr  = alloc1(maxrowline); w.cci  = alloc1(maxrowline);
        w.eer  = alloc1(maxrowline); w.eei  = alloc1(maxrowline);
        w.kkr  = alloc1(maxrowline); w.kki  = alloc1(maxrowline);
        w.mfr  = alloc1(maxrowline); w.mfi  = alloc1(maxrowline);
        w.i11r = alloc1(maxrowline); w.i11i = alloc1(maxrowline);
        w.i22r = alloc1(maxrowline); w.i22i = alloc1(maxrowline);
        w.i12r = alloc1(maxrowline); w.i12i = alloc1(maxrowline);
        w.v11r = alloc1(maxrowline); w.v11i = alloc1(maxrowline);
        w.v22r = alloc1(maxrowline); w.v22i = alloc1(maxrowline);
        w.v12r = alloc1(maxrowline); w.v12i = alloc1(maxrowline);

        auto add = [&](const string &name, const double *ar, const double *ai,
                       const double *br, const double *bi) {
            Channel c;
            c.name = name;
            c.A_re = ar; c.A_im = ai; c.B_re = br; c.B_im = bi;
            c.ct = alloc2(klast, maxrowline);       c.ct_im = alloc2(klast, maxrowline);
            c.ct_re_im = alloc2(klast, maxrowline); c.ct_im_re = alloc2(klast, maxrowline);
            c.acc_re = alloc2(klast, maxrowline);   c.acc_im = alloc2(klast, maxrowline);
            w.ch.push_back(c);
        };

        if (have_stress) {
            add("s12s12", w.s12r, w.s12i, w.s12r, w.s12i);
            add("s11s11", w.s11r, w.s11i, w.s11r, w.s11i);
            add("s22s22", w.s22r, w.s22i, w.s22r, w.s22i);
            add("s11s22", w.s11r, w.s11i, w.s22r, w.s22i);
        }
        if (have_parts) {
            /* ideal (kinetic) part */
            add("id12id12", w.i12r, w.i12i, w.i12r, w.i12i);
            add("id11id11", w.i11r, w.i11i, w.i11r, w.i11i);
            add("id22id22", w.i22r, w.i22i, w.i22r, w.i22i);
            add("id11id22", w.i11r, w.i11i, w.i22r, w.i22i);
            /* virial (excess) part */
            add("vir12vir12", w.v12r, w.v12i, w.v12r, w.v12i);
            add("vir11vir11", w.v11r, w.v11i, w.v11r, w.v11i);
            add("vir22vir22", w.v22r, w.v22i, w.v22r, w.v22i);
            add("vir11vir22", w.v11r, w.v11i, w.v22r, w.v22i);
        }
        if (have_cqch) {
            add("cc", w.ccr, w.cci, w.ccr, w.cci);
            add("ee", w.eer, w.eei, w.eer, w.eei);
            if (have_stress) {
                add("s11c", w.s11r, w.s11i, w.ccr, w.cci);
                add("s22c", w.s22r, w.s22i, w.ccr, w.cci);
                add("s11e", w.s11r, w.s11i, w.eer, w.eei);
                add("s22e", w.s22r, w.s22i, w.eer, w.eei);
            }
        }
        if (have_psiq) {
            add("kk", w.kkr, w.kki, w.kkr, w.kki);
            if (have_stress) {
                add("s11k", w.s11r, w.s11i, w.kkr, w.kki);
                add("s22k", w.s22r, w.s22i, w.kkr, w.kki);
            }
        }
    };

    {   /* print channel list once */
        Workspace tmp; build_workspace(tmp);
        printf("Active channels:");
        for (auto &c : tmp.ch) printf(" %s", c.name.c_str());
        printf("\n");
    }

#ifdef _OPENMP
    printf("OpenMP: %d threads\n", omp_get_max_threads());
#else
    printf("OpenMP: not enabled (compile with -fopenmp)\n");
#endif

    int shells_done = 0;
    time_t t_start = time(NULL);
    draw_progress(0, nqqq, (double)t_start);

    /* ---- parallel loop over shells ----
       Shared read-only: tot, qch, psq, timevect, qqq.
       Everything mutable lives in the per-thread Workspace.           */
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        Workspace w;
        build_workspace(w);

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for (int qv = 0; qv < nqqq; qv++) {

            for (auto &c : w.ch) {
                zeros(klast, nlast, ms, Nf, c.acc_re, c.acc_im, c.ct_re_im, c.ct_im_re);
                zeros(klast, nlast, ms, Nf, c.ct, c.ct_im, c.ct_re_im, c.ct_im_re);
            }

            double dqcnt = 0;

            auto process = [&](int qx, int qy) {
                if (qqq[qv] != qx * qx + qy * qy) return;
                int vect = find_index(qx, qy, qmax, QN);
                if (vect < 0) return;        /* not in the explicit q-list */
                dqcnt += 1;

                double r = sqrt((double)(qx * qx + qy * qy));
                double nx = qx / r, ny = qy / r;
                double mx = ny, my = (nx == 0) ? 0 : -nx;

                for (int f = 0; f < Nf; f++) {
                    if (have_stress) {
                        w.s11r[f] = tot[2][vect][f]*nx*nx + 2*tot[0][vect][f]*nx*ny + tot[4][vect][f]*ny*ny;
                        w.s11i[f] = tot[3][vect][f]*nx*nx + 2*tot[1][vect][f]*nx*ny + tot[5][vect][f]*ny*ny;
                        w.s22r[f] = tot[2][vect][f]*mx*mx + 2*tot[0][vect][f]*mx*my + tot[4][vect][f]*my*my;
                        w.s22i[f] = tot[3][vect][f]*mx*mx + 2*tot[1][vect][f]*mx*my + tot[5][vect][f]*my*my;
                        w.s12r[f] = tot[2][vect][f]*nx*mx + tot[0][vect][f]*(nx*my+ny*mx) + tot[4][vect][f]*ny*my;
                        w.s12i[f] = tot[3][vect][f]*nx*mx + tot[1][vect][f]*(nx*my+ny*mx) + tot[5][vect][f]*ny*my;
                    }
                    if (have_parts) {
                        /* ideal part from kin[1..6]; virial part from exc[1..6] */
                        double kxy=kin[1][vect][f], kxyi=kin[2][vect][f], kxx=kin[3][vect][f], kxxi=kin[4][vect][f], kyy=kin[5][vect][f], kyyi=kin[6][vect][f];
                        double exy=exc[1][vect][f], exyi=exc[2][vect][f], exx=exc[3][vect][f], exxi=exc[4][vect][f], eyy=exc[5][vect][f], eyyi=exc[6][vect][f];
                        w.i11r[f]=kxx*nx*nx+2*kxy*nx*ny+kyy*ny*ny; w.i11i[f]=kxxi*nx*nx+2*kxyi*nx*ny+kyyi*ny*ny;
                        w.i22r[f]=kxx*mx*mx+2*kxy*mx*my+kyy*my*my;       w.i22i[f]=kxxi*mx*mx+2*kxyi*mx*my+kyyi*my*my;
                        w.i12r[f]=kxx*nx*mx+kxy*(nx*my+ny*mx)+kyy*ny*my; w.i12i[f]=kxxi*nx*mx+kxyi*(nx*my+ny*mx)+kyyi*ny*my;
                        w.v11r[f]=exx*nx*nx+2*exy*nx*ny+eyy*ny*ny;       w.v11i[f]=exxi*nx*nx+2*exyi*nx*ny+eyyi*ny*ny;
                        w.v22r[f]=exx*mx*mx+2*exy*mx*my+eyy*my*my;       w.v22i[f]=exxi*mx*mx+2*exyi*mx*my+eyyi*my*my;
                        w.v12r[f]=exx*nx*mx+exy*(nx*my+ny*mx)+eyy*ny*my; w.v12i[f]=exxi*nx*mx+exyi*(nx*my+ny*mx)+eyyi*ny*my;
                    }
                    if (have_cqch) {
                        w.ccr[f] = qch[1][vect][f];
                        w.cci[f] = qch[2][vect][f];
                        w.eer[f] = qch[3][vect][f];
                        w.eei[f] = qch[4][vect][f];
                    } else if (have_psiq) {
                        w.ccr[f] = psq[1][vect][f];
                        w.cci[f] = psq[2][vect][f];
                    }
                    if (have_psiq) {
                        w.kkr[f] = psq[3][vect][f];
                        w.kki[f] = psq[4][vect][f];
                    }
                }

                if (have_muF) {
                    /* Connected variance  <|s|^2> - |<s>|^2  of a shear-type
                       stress component.  mu_F is the FLUCTUATION part of the
                       modulus: mu_F = mu_0 - mu_1 = (V/T)[<s^2> - <s>^2],
                       which enters mu = mu_A - mu_F.  In 2D there are two
                       independent pure-shear channels, at 0 and 45 degrees;
                       they are equal for an isotropic system, so the mean of
                       the two is the lower-noise estimator (*_ori).
                         muF_rot : 1/2(s11 - s22)  in the q-rotated frame
                         muF0    : sxy             in the frame  (0 deg)
                         muF45   : 1/2(sxx - syy)  in the frame  (45 deg)   */
                    auto connected_var = [&](const double *re, const double *im) {
                        double s0 = 0.0, s1r = 0.0, s1i = 0.0;
                        for (int f = 0; f < Nf; f++) {
                            s0  += re[f] * re[f] + im[f] * im[f];
                            s1r += re[f];
                            s1i += im[f];
                        }
                        s0 /= Nf; s1r /= Nf; s1i /= Nf;
                        return s0 - (s1r * s1r + s1i * s1i);
                    };

                    /* rotated: 1/2 (s11 - s22) */
                    for (int f = 0; f < Nf; f++) {
                        w.mfr[f] = 0.5 * (w.s11r[f] - w.s22r[f]);
                        w.mfi[f] = 0.5 * (w.s11i[f] - w.s22i[f]);
                    }
                    muF_acc[qv] += connected_var(w.mfr, w.mfi);

                    /*  0 deg: sigma_xy  (no rotation) */
                    for (int f = 0; f < Nf; f++) {
                        w.mfr[f] = tot[0][vect][f];
                        w.mfi[f] = tot[1][vect][f];
                    }
                    muF0_acc[qv] += connected_var(w.mfr, w.mfi);

                    /*  45 deg: 1/2 (sigma_xx - sigma_yy)  (no rotation) */
                    for (int f = 0; f < Nf; f++) {
                        w.mfr[f] = 0.5 * (tot[2][vect][f] - tot[4][vect][f]);
                        w.mfi[f] = 0.5 * (tot[3][vect][f] - tot[5][vect][f]);
                    }
                    muF45_acc[qv] += connected_var(w.mfr, w.mfi);
                }

                for (auto &c : w.ch) {
                    zeros(klast, nlast, ms, Nf, c.ct, c.ct_im, c.ct_re_im, c.ct_im_re);
                    cross_correlation(klast, nlast, ms, Nf, maxrowline,
                                      c.A_re, c.A_im, c.B_re, c.B_im,
                                      c.ct, c.ct_im, c.ct_re_im, c.ct_im_re);
                    /* C(t) = <A*(0) B(t)> of this q-vector; with absC the modulus
                       |C(t)| of s22s22 is accumulated instead (Im -> 0) */
                    const bool use_abs = want_absC && c.name == "s22s22";
                    auto accum = [&](int k, int n) {
                        double re = c.ct[k][n] + c.ct_im[k][n];
                        double im = c.ct_re_im[k][n] - c.ct_im_re[k][n];
                        if (use_abs) c.acc_re[k][n] += sqrt(re * re + im * im);
                        else { c.acc_re[k][n] += re; c.acc_im[k][n] += im; }
                    };
                    for (int n = 0; n <= 2 * ms; n++)          accum(0, n);
                    for (int k = 1; k < klast; k++)
                        for (int n = ms + 1; n <= 2 * ms; n++) accum(k, n);
                    for (int n = ms + 1; n < nlast; n++)       accum(klast, n);
                }
            };

            /* loop over the wavevectors actually present:
               the explicit list if one was given, otherwise the regular grid
               within the analysis range qlim (qmax is the file's row-layout
               parameter, not the q range)                                    */
            if (!QLIST_IL.empty()) {
                for (size_t k = 0; k < QLIST_IL.size(); k++)
                    process(QLIST_IL[k], QLIST_JM[k]);
            } else {
                for (int qx = -qlim; qx <= qlim; qx += QN)
                    for (int qy = QN; qy <= qlim; qy += QN)
                        process(qx, qy);
                for (int qx = QN; qx <= qlim; qx += QN)
                    process(qx, 0);
            }

            shell_nq[qv] = dqcnt;
            if (dqcnt == 0) continue;

            /* each shell writes its own unique file: no thread conflicts */
            /* filename labelled with the analysed range qlim, so a folder of
               Qmax<qlim> files always contains exactly q <= qlim, regardless of
               which file (dq=1 or dq=4, small or large grid) produced it */
            string name = OUTPREF + ".Qmax" + to_string(qlim) + ".q" + to_string(qqq[qv]) + ".dat";
            ofstream out(name.c_str());
            out << "# t";
            for (auto &c : w.ch) out << " " << c.name << "_re " << c.name << "_im";
            out << "\n";

            double D = dqcnt;
            auto row = [&](double tval, int k, int n) {
                out << setprecision(10) << tval;
                for (auto &c : w.ch)
                    out << " " << c.acc_re[k][n] / D << " " << c.acc_im[k][n] / D;
                out << "\n";
            };
            for (int n = 0; n <= 2 * ms; n++)          row(timevect[n] * DT, 0, n);
            for (int k = 1; k < klast; k++)
                for (int n = ms + 1; n <= 2 * ms; n++) row(pow(2, k) * timevect[n] * DT, k, n);
            for (int n = ms + 1; n < nlast; n++)       row(pow(2, klast) * timevect[n] * DT, klast, n);
            out.close();

#ifdef _OPENMP
#pragma omp atomic
#endif
            shells_done++;

            /* only one thread redraws */
            draw_progress(shells_done, nqqq, (double)t_start);
        }
    } /* end parallel */
    fprintf(stderr, "\n");

    /* ---- mu_F(q) summary ---- */
    if (have_muF) {
        string fname = OUTPREF + "_muF_Qmax" + to_string(qlim) + ".dat";
        ofstream mout(fname.c_str());
        mout << "# q2_int  muF_rot  muF0  muF45  muF_ori"
             << "   [RAW connected variances, no V/T -- multiply by V/T downstream;"
             << " muF_rot=1/2(s11-s22) rotated, muF0=sxy , muF45=1/2(sxx-syy) ,"
             << " muF_ori=(muF0+muF45)/2]\n";
        mout << setprecision(10);
        for (int qv = 0; qv < nqqq; qv++) {
            if (shell_nq[qv] == 0) continue;
            double muF_rot = muF_acc[qv]   / shell_nq[qv];   /* RAW, no V/T */
            double muF0    = muF0_acc[qv]  / shell_nq[qv];
            double muF45   = muF45_acc[qv] / shell_nq[qv];
            double muF_ori = 0.5 * (muF0 + muF45);
            mout << qqq[qv] << "  " << muF_rot << "  " << muF0
                 << "  " << muF45 << "  " << muF_ori << "\n";
        }
        mout.close();
        printf("Wrote %s (affine modulus mu_F vs q)\n", fname.c_str());
    }
    free(muF_acc); free(muF0_acc); free(muF45_acc);
    free(shell_nq);

    printf("Done.\n");
    return 0;
}
