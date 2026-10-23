# kstress

LAMMPS computes for the wavevector-dependent stress tensor σ_αβ(q) of a two-dimensional system, and a multiple-tau correlator for the stress correlation functions C(q,t).

## Contents

| file | what it is |
|---|---|
| `src/compute_kstress_virial.cpp/.h` | `compute kstress_vir`: virial (pair) part of σ(q) |
| `src/compute_kstress_kinetic.cpp/.h` | `compute kstress_kin`: kinetic (ideal) part of σ(q) |
| `tools/correlation_kstress.cpp` | correlator, independent of LAMMPS |

## Installation

```
cp src/compute_kstress_*.cpp src/compute_kstress_*.h $LAMMPS/src/
cd $LAMMPS/build && cmake --build . -j
```

## Wavevector file

One wavevector per line, two integers `il jm`, with q = (2π il/Lx, 2π jm/Ly). Blank lines and lines starting with `#` are ignored. Row k of every output is line k of the file, so the same file must be given to both computes and to the correlator.

```
# il  jm
   0   0     # q = 0: check against compute pressure
   4   0
   0   4
  -4   4
```

## Usage

```
compute ks  all kstress_vir qlist.dat ${ksp}
compute kin all kstress_kin qlist.dat ${ksp}
fix fe all ave/time 1 1 ${ksp} c_ks[*]  format "%20.11e" file kstress.virial.dat  mode vector
fix fk all ave/time 1 1 ${ksp} c_kin[*] format "%20.11e" file kstress.kinetic.dat mode vector
```

Output: one row per wavevector, columns `sxy_re sxy_im sxx_re sxx_im syy_re syy_im`.

Correlator:

```
g++ -O2 -fopenmp -o cort tools/correlation_kstress.cpp -lm
./cort 50 qlist=qlist.dat kin=kstress.kinetic.dat exc=kstress.virial.dat parts
```

`parts` also writes the ideal and virial contributions separately. One file per shell |q|^2 = il^2 + jm^2; column 1 is the lag time, then (re, im) of each channel. No V/T factor is applied.

## Restrictions

- 2D, orthogonal box. The correlator projections assume Lx = Ly.
- `newton_pair on` (the LAMMPS default). With `newton_pair off`, pairs across processor boundaries are counted twice.
- Pair forces only: no many-body potentials, no bond, angle, dihedral, improper or kspace contributions.
- The pair style must call the LAMMPS tally interface; GPU, KOKKOS and INTEL accelerated styles do not.
- Each MPI rank must contain interacting pairs at every sampled step (always the case for dense liquids and glasses); a rank without pairs would keep the values of the previous sample.
- Velocities are used as they are; no centre-of-mass motion or bias is removed.
- The correlator lag time assumes a timestep of 0.005 (constant `DT` in `correlation_kstress.cpp`).

## Checking a new system

Include `0 0` in the wavevector file and compare row 0 with the stress computed by LAMMPS:

```
compute pv all pressure thermo_temp virial   # compare with kstress.virial.dat
compute pk all pressure thermo_temp ke       # compare with kstress.kinetic.dat
```

## License

GPL-2.0, as LAMMPS.

## Citation

O. Benzerara, J. Baschnagel, A. N. Semenov, L. Klochko, *kstress: a LAMMPS package for wavevector-dependent stress correlations*, Computer Physics Communications (submitted).
