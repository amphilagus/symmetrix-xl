# MACE-POLAR

`SymmetrixPolar` is an experimental mixed native/PyTorch ASE calculator for the
one-interaction nonlinear scalar backbone used by MACE-POLAR-1-S. It evaluates
the backbone features and their vector-Jacobian product in Symmetrix's FP64
Kokkos evaluator. The original POLAR modules evaluate the charge/spin sources,
two field updates, reciprocal-space electrostatics, and local electron energy
in PyTorch. The original model's energy readout and scale/shift are retained.

This interface requires PyTorch, e3nn, MACE with `PolarMACE`, and
`graph_longrange`, as well as a matching Symmetrix native build. Unlike ordinary
compact JSON inference, it needs the whole POLAR checkpoint and its Python
runtime. The initial implementation targets the pinned MACE 0.3.16 and
graph-longrange 0.4.4 runtime. Those versions require the maintained source
compatibility corrections for the graph API; installing the version numbers
alone does not establish compatibility. Existing whole-model checkpoints may
also require restoration of missing derived graph metadata. The adapter audits
that the learned tensors, parameters, buffers, and module objects remain
unchanged during that restoration.

## Using the calculator

```python
from ase.io import read
from symmetrix.polar import SymmetrixPolar

atoms = read("lithium-water.extxyz")
atoms.info["charge"] = 1.0
atoms.info["spin"] = 1.0
atoms.info["external_field"] = [0.0, 0.0, 0.0]
atoms.calc = SymmetrixPolar(
    "/absolute/path/MACE-POLAR-1-S.model",
    device="cpu",
    bridge="host",
    precision="float64",
)
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
charges = atoms.calc.results["charges"]
diagnostics = atoms.calc.results["polar_diagnostics"]
```

The first argument can also be a `PolarMACE` module. It is copied before use.
Use a trusted checkpoint: loading a whole PyTorch model uses Python pickle.

`charge` and `spin` constructor arguments provide defaults when the matching
`atoms.info` entries are absent; the defaults are zero charge and spin
multiplicity one. `atoms.info` takes precedence. Charge must be finite, and
spin is a positive integer multiplicity, so the constrained spin population is
`spin - 1`. Changing either value invalidates the ASE calculation cache.

The results include `energy`, `free_energy`, `forces`, `charges`,
`density_coefficients`, `spin_charge_density`, `electrostatic_energy`, and
`electron_energy`. The total energy contains the original atomic and local
neural contributions, electrostatic energy, and local electron energy exactly
once. Adding an independent Coulomb calculator would change the model.

## Execution boundary

The initial supported contract is:

- One nonlinear residual interaction with a scalar feature output and a linear
  energy readout, as used by MACE-POLAR-1-S; two POLAR field updates.
- FP64 on the selected Kokkos CPU or CUDA backend, with the PyTorch suffix on
  the same device. Native CPU and CUDA libraries require separate processes
  and separately qualified builds.
- One structure with full three-dimensional periodic boundary conditions and
  a nonsingular cell. Charge and spin use the original model's constraints.
- Zero external electric field and the original periodic reciprocal-space
  convention, including removal of the zero wave-vector contribution.
- Energies, forces, and the density outputs above. Stress, FP32, isolated or
  partially periodic structures, MPI/LAMMPS integration, and larger/deeper
  POLAR architectures are outside this implementation's admission boundary.

Unsupported requests fail explicitly. Ordinary `symmetrix_extract_mace` rejects
a whole `PolarMACE` checkpoint and directs callers to `SymmetrixPolar`: exporting
only its MACE-like submodules would discard long-range terms.

`bridge="host"` exports feature snapshots as NumPy arrays and stages adjoints
back into the native evaluator. It is useful for CPU execution and debugging.
Omitting `bridge`, or setting `bridge="auto"`, selects host exchange on CPU
and DLPack on CUDA. The native and PyTorch devices must match, including the
CUDA device ordinal; a CPU prefix is rejected for a requested CUDA calculation.
`bridge="dlpack"` exports device-owned feature snapshots and copies adjoint
seeds between device allocations through DLPack. It keeps CUDA force reduction
on the device. This is a device snapshot and device-to-device copy bridge.
Snapshots own their
storage; they do not alias a later native calculation. The current bridge uses
explicit synchronization when exchanging tensors between Kokkos and PyTorch.
It does not establish a zero-overhead or asynchronous execution guarantee.

For one diagnostic call, `profile=True` adds synchronized stage timings in
`results["polar_diagnostics"]["timings_ms"]`. It separates graph construction,
native feature evaluation/export, the PyTorch suffix and autograd, native VJP,
and force/result assembly. Throughput measurements must keep `profile=False`,
which is the default; synchronized diagnostic calls are separate from timing
samples.

The native prefix currently uses materialized nonlinear execution. It does not
claim the prepared direct, generated R1, or fixed-workspace performance of
ordinary MACE. Timing must compare complete POLAR energy/force calculations
against the original complete model, after both runtimes have warmed up.

## Qualification and provenance

The `polar_diagnostics` result reports the feature path, bridge, precision, native class
and build information, feature shapes, POLAR recursion count, boundary
convention, charge/spin normalization, and energy decomposition. A successful
import alone does not qualify a mixed runtime. Record the Symmetrix source and
native-content hashes, loaded extension path/hash, checkpoint hash, Python,
PyTorch, MACE, e3nn, graph-longrange versions, compatibility patch identity,
selected Kokkos execution space, and CUDA runtime versions where applicable.
Do not mix a Symmetrix CUDA extension with arbitrary library paths from the
PyTorch environment to make imports succeed.

The reduced oracle tests build a real small `PolarMACE` with Li/H/O, then compare
the full energy, force, density, and conserved populations on a charged Li plus
four-water periodic structure. Separate tests compare actual per-layer native
features and arbitrary feature-adjoint VJPs with upstream blocks; they also
check DLPack ownership, stale tape rejection, translation, periodic wrapping,
coordinate perturbations, charge/spin changes, and a force finite difference.

Run these tests in each freshly built CPU/CUDA environment:

```bash
pytest -q symmetrix/test/test_polar.py symmetrix/test/test_polar_features.py
```

The real 512-channel foundation-model oracle is opt-in and never downloads the
checkpoint:

```bash
SYMMETRIX_POLAR_MODEL=/absolute/path/MACE-POLAR-1-S.model \
    pytest -q symmetrix/test/test_polar.py::test_real_h512_s_checkpoint_oracle
```

A missing optional model skips this test; an explicitly configured nonexistent
path fails. Report checkpoint availability and skipped tests alongside parity
and performance results.

## Li/water deployment experiments

The reproducible driver is `benchmarks/polar_li_water.py`. Each prepared box
contains exactly one Li and 64, 128, 256, or 512 complete waters, with total
charge +1, spin multiplicity one, zero external field, full PBC, and water
density 1 g/cm³. Larger boxes are independently constructed; a charged box is
never repeated. Preparation uses fixed seeds and fixed-cell original-model FP64
optimization with at most 2000 steps and `fmax=0.05 eV/Å`, followed by 500 NVT steps
at 300 K and 0.25 fs. Failure to converge is recorded and stops preparation.
`--optimizer auto` uses LBFGS for at least 256 waters and BFGS for smaller boxes;
`--optimizer bfgs` and `--optimizer lbfgs` select either method explicitly. The
manifest records the actual optimizer class, and its log/trajectory suffix is
`.bfgs` or `.lbfgs`. Resuming with `--input` preserves the convergence criterion
and cell; changing optimizer does not imply a new thermodynamic state.

The deployed `deployment/polar/experiment.sbatch` wrapper takes the deployment
root, runtime profile (`cpu`, `sm80`, or `sm90`), then a driver subcommand. It
selects the profile's interpreter and backend and supplies the local checkpoint,
device, and bridge arguments. Submit model calculations through Slurm. CPU
allocation defaults come from the wrapper; GPU submissions must request a GPU
matching the profile through the site's partition/resource options.

From the deployed repository, for example:

```bash
DEPLOYMENT_ROOT=/absolute/path/polar-deployment
OUTPUT_ROOT=/absolute/path/polar-results

# A CPU preparation example; initial generation can be selected independently.
sbatch deployment/polar/experiment.sbatch "$DEPLOYMENT_ROOT" cpu \
    prepare --stage initial --waters 64 128 256 512 \
    --output "$OUTPUT_ROOT/prepared"

# Set GPU_PARTITION according to the cluster and select matching sm80 hardware.
sbatch --partition="$GPU_PARTITION" --gres=gpu:1 \
    deployment/polar/experiment.sbatch "$DEPLOYMENT_ROOT" sm80 \
    prepare --stage all --waters 64 --output "$OUTPUT_ROOT/prepared"

sbatch --partition="$GPU_PARTITION" --gres=gpu:1 \
    deployment/polar/experiment.sbatch "$DEPLOYMENT_ROOT" sm80 \
    validate --input "$OUTPUT_ROOT/prepared/water0064.equilibrated.extxyz" \
    --output "$OUTPUT_ROOT/sm80-water0064/validation"

sbatch --partition="$GPU_PARTITION" --gres=gpu:1 \
    deployment/polar/experiment.sbatch "$DEPLOYMENT_ROOT" sm80 \
    md --steps 100 --sample-interval 10 \
    --input "$OUTPUT_ROOT/prepared/water0064.equilibrated.extxyz" \
    --output "$OUTPUT_ROOT/sm80-water0064/nve100"

sbatch --partition="$GPU_PARTITION" --gres=gpu:1 \
    deployment/polar/experiment.sbatch "$DEPLOYMENT_ROOT" sm80 \
    benchmark --repeats 3 --seconds 10 \
    --input "$OUTPUT_ROOT/prepared/water0064.equilibrated.extxyz" \
    --output "$OUTPUT_ROOT/sm80-water0064/benchmark"
```

Repeat the accepted procedure for each water count and runtime profile. Run
`md --steps 1000` after the shorter trajectory passes. `prepare --stage optimize`
and `prepare --stage equilibrate` support separate jobs; an explicit `--input`
resumes one water count. Equilibration verifies an explicitly resumed structure's
force convergence. Without `--input`, it requires the stored converged
optimization record and the same checkpoint hash.

Validation compares complete original and candidate energies/forces on the
base, translated, and periodically wrapped structures. The default acceptance
thresholds are absolute total-cell energy error ≤ `1e-4 eV`, maximum force
component error ≤ `1e-5 eV/Å`, charge error ≤ `1e-9 e`, and spin-population error
≤ `1e-9`, with the last two checked before and after both field updates. Force
errors are reported separately for Li, all three atoms of hydration-shell waters
whose oxygen lies within 3.2 Å of Li using the minimum-image distance, and the
remaining waters. Region membership is recomputed on each sampled frame.

NVE uses the prepared positions and momenta, preserving the equilibrated state.
Only inputs without momenta receive deterministic Maxwell velocities; the
fallback is recorded. Preparation records the NVT initial/final state hashes
and hashes of the actual written-and-reloaded output. NVE and each timing worker
record their initial position and velocity hashes, so the prepared state and
original/candidate starting conditions can be audited. Same-trajectory NVE
checks recompute the original energy and forces on sampled candidate frames
and report total-energy drift, maximum drift per atom, and its linear slope.
No numerical drift acceptance threshold is implied by the parity thresholds.

Timing uses three independent processes per executor and complete ASE NVE
blocks lasting at least ten seconds after warmup, including neighbor creation,
force transfers, and the final device synchronization. Before importing NumPy,
BLAS, PyTorch, or Kokkos, the driver pins itself and its children to one allowed
logical CPU, excludes its SMT siblings, and sets worker counts to one. It records
the selected logical CPU and Linux socket/core/sibling topology. Candidate
throughput uses `profile=False` and omits diagnostic histories; one separately
synchronized `profile=True` calculation provides stage times outside the timing
and memory samples.

JSON records include input/checkpoint hashes, Git identity, package and source
fingerprints, actual executor and native build information, atom/water counts,
cutoff, directed edges, precision, `us/atom/step`, and memory evidence. GPU
polling prefers the process's total NVIDIA allocation, covering both Torch and
native allocators. It also records all-GPU baseline/peak allocation and competing
process IDs; global figures include other users' allocations. Torch allocator
peaks are labeled separately. Missing counters remain unavailable. `report`
summarizes worker JSON files only after checking that their physical inputs,
protocols, paired initial states, and repeat identities agree. All result paths
are explicitly selected with `--output`; generated data are not source artifacts.

The paired GPU pipeline `deployment/polar/qualify_gpu_pair.sbatch` runs both
profiles in one allocation containing three GPUs, including an SM80 and an
SM90. Deploy the reviewed source and matching profile builds first, then submit:

```bash
sbatch --output="$DEPLOYMENT_ROOT/logs/polar-gpu-pair-%j.out" \
    deployment/polar/qualify_gpu_pair.sbatch "$DEPLOYMENT_ROOT"
```

The wrapper defaults to the `compute` partition, three GPUs, four logical CPUs,
192 GB RAM, and 24 hours; site scheduler options can be overridden on the
`sbatch` command line. It selects only devices already listed in Slurm's
`CUDA_VISIBLE_DEVICES`, preserves each selected original token, and gives each
profile a single visible GPU at logical ordinal zero. Two distinct physical
CPU cores are required; their SMT siblings are excluded. The profiles run
concurrently with separate CPU affinities, GPUs, interpreters, and backends.

The default inputs are `benchmark/prepared-reference/waterXXXX.prepare.json`
and `.equilibrated.extxyz` under the deployment root. All four preparations
must have passed optimization and 500-step NVT, use the same checkpoint, and
match their recorded file and velocity hashes. Each profile runs the four
geometry validations, then 64-water NVE for 100 and 1000 steps, then all four
sizes with three independent original and candidate timing workers per size,
each timed for at least ten seconds. Both NVE runs start from the same prepared
state. Energy, force, regional-force, and initial/two-update charge/spin
thresholds remain those above. Drift is recorded without a new drift threshold.
The pipeline verifies actual native Cuda execution and FP64 DLPack features.

Results use `benchmark/qualification/<profile>/waterXXXX/{validation,md100,md1000}`
and `benchmark/performance/<profile>/waterXXXX/summary.json`; only the 64-water
case has the two MD directories. GPU selection, physical CPU topology, command
logs, planned and completed stages, and failures are saved in
`benchmark/gpu-pair/<job-id>`. A failed stage stops its profile; the other
profile can finish, and the allocation returns a failure status. Existing
nonempty result directories are rejected. A deliberate rerun can pass
`--output-root /absolute/path/new-results`; `--prepared` and `--checkpoint`
also accept explicit paths. No stages are skipped or marked passed from an
older run.

`benchmarks/polar_qualification_report.py` reads deployment evidence without
running models or tests. Its `--output` must be outside a Git checkout. The
report prefers the separately built `cpu-omp` profile for formal CPU results
and requires `manifests/build/cpu-omp/doctor-torch-polar.json` to show a strict
`ok` result after real Torch/native POLAR energy and force evaluation. Older
`cpu` trajectories and diagnostic timings remain labeled separately. The
report preserves full-suite failures and skips, compares matching failures
with a pristine upstream baseline when available, and distinguishes archived
JUnit attempts from each profile's current qualification run.

CPU MD correctness runs can explicitly use `md --cpu-threads 8` when the
allocation provides eight physical cores. The driver applies affinity before
numerical imports, excludes SMT siblings, uses eight Kokkos/OpenMP threads and
one BLAS thread, and verifies actual native concurrency. GPU calculations and
all formal `benchmark`/`_worker` throughput runs require one CPU thread.
