#!/bin/bash
#SBATCH --job-name=polar_gpu_pair
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=24:00:00
set -euo pipefail

root=${1:?Usage: qualify_gpu_pair.sbatch DEPLOYMENT_ROOT [controller options]}
shift
export PATH="$root/opt/sm80/bin:$(dirname "$root")/mace-symmetrix-xl/opt/git/bin:/usr/bin:/bin"
export LC_ALL=C LANG=C
export OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$root/src/symmetrix-xl"
git rev-parse HEAD
exec srun --cpu-bind=cores "$root/opt/sm80/bin/python" \
    deployment/polar/qualify_gpu_pair.py --root "$root" "$@"
