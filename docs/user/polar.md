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
`bridge="dlpack"` transfers device-owned feature and adjoint tensors through
DLPack, and keeps the CUDA force reduction on the device. Snapshots own their
storage; they do not alias a later native calculation. The current bridge uses
explicit synchronization when exchanging tensors between Kokkos and PyTorch.
It does not establish a zero-overhead or asynchronous execution guarantee.

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
