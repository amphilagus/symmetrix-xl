#!/bin/bash
#SBATCH --job-name=polar_qualify
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=24:00:00
set -euo pipefail
root=$1
profile=$2
shift 2
export PATH="$root/opt/$profile/bin:/usr/bin:/bin"
export LC_ALL=C LANG=C
export OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SYMMETRIX_BACKEND=cpu
device=cpu
bridge=host
if [[ "$profile" != cpu ]]; then
    export SYMMETRIX_BACKEND="cuda12-$profile"
    device=cuda
    bridge=dlpack
    nvidia-smi
fi
cd "$root/src/symmetrix-xl"
git rev-parse HEAD
exec srun --cpu-bind=cores "$root/opt/$profile/bin/python" \
    benchmarks/polar_li_water.py "$@" \
    --checkpoint "$root/forcefields/polar1s/MACE-POLAR-1-S.model" \
    --device "$device" --bridge "$bridge"
