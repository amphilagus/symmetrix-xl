#!/bin/bash
#SBATCH --job-name=polar_gpu_profile
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=01:00:00
set -euo pipefail
root=$1
source_dir="$root/src/symmetrix-xl"
output="$root/benchmark/module-profile/$SLURM_JOB_ID"
mkdir -p "$root/benchmark/module-profile"
mkdir -m 700 "$output"
unset PYTHONHOME PYTHONPATH LD_PRELOAD LD_AUDIT
export LD_LIBRARY_PATH="$root/opt/base/gcc-runtime/lib64"
export PATH="$(dirname "$root")/mace-symmetrix-xl/opt/git/bin:/usr/bin:/bin"
export LC_ALL=C LANG=C PYTHONNOUSERSITE=1 NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OMP_MAX_ACTIVE_LEVELS=1
nsys="$(dirname "$root")/opt/nvhpc/Linux_x86_64/24.3/profilers/Nsight_Systems/bin/nsys"
"$nsys" --version > "$output/nsys-version.txt"
nvidia-smi > "$output/nvidia-smi.txt"
cd "$source_dir"
git rev-parse HEAD > "$output/source-commit.txt"
original_visibility=$CUDA_VISIBLE_DEVICES
for profile in sm80 sm90; do
    python="$root/opt/$profile/bin/python"
    selected_gpu=$(CUDA_VISIBLE_DEVICES="$original_visibility" "$python" \
        deployment/polar/select_device.py --profile "$profile" \
        2> "$output/$profile-selection.jsonl")
    export CUDA_VISIBLE_DEVICES="$selected_gpu" SYMMETRIX_BACKEND="cuda12-$profile"
    srun --cpu-bind=cores env CUDA_VISIBLE_DEVICES="$selected_gpu" \
        SYMMETRIX_BACKEND="cuda12-$profile" "$python" benchmarks/polar_module_profile.py \
        --checkpoint "$root/forcefields/polar1s/MACE-POLAR-1-S.model" \
        --input "$root/benchmark/prepared-reference/water0064.equilibrated.extxyz" \
        --device cuda:0 --bridge dlpack --warmup 2 --calls 3 \
        --output "$output/$profile-modules.json" \
        > "$output/$profile-modules.log" 2>&1

    # The normal wrapper clears inherited LD_PRELOAD. Launch the exact same
    # interpreter/loader paths directly so Nsight's injection survives. Cache
    # paths and numerical settings stay confined to this deployment/profile.
    library_path=$("$python" -c 'from pathlib import Path; import sys; sys.path.insert(0,"deployment/polar"); from bootstrap import library_paths; print(":".join(library_paths(Path(sys.argv[1]),sys.argv[2])))' "$root" "$profile")
    export XDG_CACHE_HOME="$root/build/cache/$profile/xdg"
    export CUDA_CACHE_PATH="$root/build/cache/$profile/cuda"
    export TORCH_EXTENSIONS_DIR="$root/build/cache/$profile/torch"
    export CUEQUIVARIANCE_CACHE_DIR="$root/build/cache/$profile/cueq"
    export SYMMETRIX_JIT_CACHE="$root/build/cache/$profile/symmetrix"
    srun --cpu-bind=cores env CUDA_VISIBLE_DEVICES="$selected_gpu" \
        SYMMETRIX_BACKEND="cuda12-$profile" \
        "$nsys" profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        --output="$output/$profile-normal-call" \
        "$root/opt/base/glibc-2.28/lib/ld-2.28.so" --library-path "$library_path" \
        "$root/opt/$profile/bin/python3.real" benchmarks/polar_module_profile.py \
        --checkpoint "$root/forcefields/polar1s/MACE-POLAR-1-S.model" \
        --input "$root/benchmark/prepared-reference/water0064.equilibrated.extxyz" \
        --device cuda:0 --bridge dlpack --warmup 2 --calls 1 \
        --cuda-profiler-range --output "$output/$profile-traced-modules.json" \
        > "$output/$profile-nsys.log" 2>&1
    "$nsys" stats --format csv \
        --report cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
        "$output/$profile-normal-call.nsys-rep" \
        > "$output/$profile-cuda-statistics.txt" 2>&1
done
