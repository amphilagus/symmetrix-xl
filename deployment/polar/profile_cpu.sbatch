#!/bin/bash
#SBATCH --job-name=polar_cpu_profile
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=00:30:00
set -euo pipefail
root=$1
profile=cpu-omp
output="$root/benchmark/module-profile/$SLURM_JOB_ID"
mkdir -p "$root/benchmark/module-profile"
mkdir -m 700 "$output"
export PATH="$(dirname "$root")/mace-symmetrix-xl/opt/git/bin:/usr/bin:/bin"
export LC_ALL=C LANG=C SYMMETRIX_BACKEND=cpu
export OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$root/src/symmetrix-xl"
git rev-parse HEAD > "$output/source-commit.txt"
exec srun --cpu-bind=cores "$root/opt/$profile/bin/python" \
    benchmarks/polar_module_profile.py \
    --checkpoint "$root/forcefields/polar1s/MACE-POLAR-1-S.model" \
    --input "$root/benchmark/prepared-reference/water0064.equilibrated.extxyz" \
    --device cpu --bridge host --warmup 2 --calls 3 \
    --output "$output/$profile-modules.json"
