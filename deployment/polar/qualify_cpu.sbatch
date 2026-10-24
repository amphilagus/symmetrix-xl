#!/bin/bash
#SBATCH --job-name=polar_cpu_qualification
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
set -euo pipefail
root=$1
export LC_ALL=C LANG=C
export PATH="$root/opt/cpu-omp/bin:$(dirname "$root")/mace-symmetrix-xl/opt/git/bin:/usr/bin:/bin"
export SYMMETRIX_BACKEND=cpu SYMMETRIX_OPENMP_RUNTIME_CHECK=strict
export OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec srun --cpu-bind=cores "$root/opt/cpu-omp/bin/python" \
    "$root/src/symmetrix-xl/deployment/polar/qualify_cpu.py" --root "$root"
