"""ASE inference with native local features and the complete MACE-POLAR suffix.

The suffix follows ACEsuit/mace v0.3.16 ``modules/extensions.py`` and
``modules/field_blocks.py`` (MIT; Ilyes Batatia and William J. Baldwin), commit
4d2da09413ac1407f37cdbb6b81fa28e4c15655e. It uses the public graph_longrange
0.4.4 API. Its copyright/permission notice is in ``_licenses/mace-MIT.txt``.
The local interaction/product modules are never called here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np
from ase.calculators.calculator import Calculator, PropertyNotImplementedError
from ase.neighborlist import neighbor_list

try:
    import torch
except ImportError as error:
    raise ImportError("SymmetrixPolar requires the symmetrix-xl[polar] extra") from error

from . import load_backend
from ._polar_compat import repair_polar_checkpoint


def _is_polar(model):
    return any(cls.__name__ == "PolarMACE" for cls in type(model).__mro__)


def _validate_model(model):
    """Reject unsupported graphs before building a partial local artifact."""
    from e3nn import o3
    from mace.modules.blocks import (
        LinearReadoutBlock,
        RealAgnosticResidualNonLinearInteractionBlock,
    )
    from mace.modules.field_blocks import (
        AgnosticEmbeddedOneBodyVariableUpdate,
        OneBodyMLPFieldReadout,
    )

    if not _is_polar(model):
        raise TypeError("SymmetrixPolar requires a complete PolarMACE checkpoint")
    if len(model.interactions) != 1 or len(model.products) != 1:
        raise ValueError("The first Polar interface supports one-interaction S models")
    if not isinstance(
        model.interactions[0], RealAgnosticResidualNonLinearInteractionBlock
    ):
        raise ValueError("Polar S requires the nonlinear residual interaction block")
    irreps = o3.Irreps(model.products[0].linear.irreps_out)
    if any(ir != o3.Irrep("0e") for _, ir in irreps):
        raise ValueError("The first Polar interface requires scalar product features")
    if len(model.readouts) != 1 or type(model.readouts[0]) is not LinearReadoutBlock:
        raise ValueError("Polar S requires its single original linear energy readout")
    if len(model.lr_source_maps) != 1 or int(model.num_recursion_steps) != 2:
        raise ValueError("Polar S requires one local source and two field updates")
    if len(model.field_dependent_charges_maps) != 2 or any(
        type(block) is not AgnosticEmbeddedOneBodyVariableUpdate
        for block in model.field_dependent_charges_maps
    ):
        raise ValueError("Unsupported Polar field update module")
    if type(model.local_electron_energy) is not OneBodyMLPFieldReadout:
        raise ValueError("Unsupported Polar non-local energy readout")
    if hasattr(model, "joint_embedding") or hasattr(model, "embedding_readout"):
        raise ValueError("Polar joint embeddings are not supported by this interface")
    if len(getattr(model, "heads", ["Default"])) != 1:
        raise ValueError("The first Polar interface requires a single-head checkpoint")


def _normalized_fukui(values):
    norm = values.double().sum(dim=0, keepdim=True).to(values.dtype)
    return values / torch.where(norm == 0, torch.ones_like(norm), norm)


def _equilibrate(density, fukui, targets):
    result = density.clone()
    deficit = targets - density[:, :, 0].double().sum(dim=0).to(density.dtype)
    result[:, :, 0] = density[:, :, 0] + fukui * deficit
    return result


def _suffix(model, features, positions, cell, node_attrs, edge_index, vectors, charge, spin):
    """Evaluate all energy readouts and charge updates using native leaf features."""
    from graph_longrange.kspace import compute_k_vectors_flat
    from mace.modules.utils import compute_total_charge_dipole_permuted

    count = positions.shape[0]
    device, dtype = positions.device, positions.dtype
    batch = torch.zeros(count, device=device, dtype=torch.long)
    node_heads = batch
    atom_index = torch.arange(count, device=device)
    pbc = torch.ones((1, 3), device=device, dtype=torch.bool)
    volume = torch.linalg.det(cell).abs().reshape(1)
    reciprocal = (2 * math.pi * torch.linalg.inv(cell).T).reshape(1, 3, 3)
    lengths = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    edge_attrs = model.spherical_harmonics(vectors[:, [1, 2, 0]])
    edge_feats, _ = model.radial_embedding(
        lengths, node_attrs, edge_index, model.atomic_numbers
    )
    node_e0 = model.atomic_energies_fn(node_attrs)[atom_index, node_heads]
    local_readouts = []
    density = positions.new_zeros((count, model.charges_irreps.dim))
    for layer, (feature, source) in enumerate(zip(features, model.lr_source_maps)):
        readout_index = -1 if len(model.readouts) == 1 else min(
            layer, len(model.readouts) - 1
        )
        local_readouts.append(
            model.readouts[readout_index](feature, node_heads)[atom_index, node_heads]
        )
        density = density + source(feature).squeeze(-2)
    node_interaction = torch.stack(local_readouts).sum(dim=0)
    node_interaction = model.scale_shift(node_interaction, node_heads)
    local_energy = node_e0.sum() + node_interaction.sum()
    if hasattr(model, "pair_repulsion"):
        # Upstream PolarMACE computes this auxiliary value but does not add it
        # to node_es_list. Preserve that pinned energy definition exactly.
        model.pair_repulsion_fn(lengths, node_attrs, edge_index, model.atomic_numbers)

    k_vectors, k_norm2, k_batch, k0_mask = compute_k_vectors_flat(
        model.kspace_cutoff, cell.reshape(1, 3, 3), reciprocal
    )
    descriptor = model.electric_potential_descriptor
    cache = descriptor.precompute_geometry(
        k_vectors=k_vectors,
        k_norm2=k_norm2,
        k_vector_batch=k_batch,
        k0_mask=k0_mask,
        node_positions=positions,
        batch=batch,
        volume=volume,
        pbc=pbc,
    )
    mixed = model.layer_feature_mixer(torch.stack(features, dim=0))
    density = density.reshape(count, 2, -1)
    fukui_input = features[-1]
    to_fukui = getattr(model, "_fukui_to_mul_ir", None)
    if to_fukui is not None:
        fukui_input = to_fukui(fukui_input)
    fukui = _normalized_fukui(model.fukui_source_map(fukui_input))
    targets = positions.new_tensor([(charge + spin - 1) / 2, (charge - spin + 1) / 2])
    density = _equilibrate(density, fukui, targets)
    initial_density = density.clone()
    histories = [density]
    fukui_histories = [fukui]
    potential_histories = []
    potential = positions.new_zeros((count, model.potential_irreps.dim))
    to_charges = getattr(model, "_charges_to_mul_ir", None)
    from_field = getattr(model, "_field_from_mul_ir", None)
    for update in model.field_dependent_charges_maps:
        fields = []
        for channel in range(2):
            source = density[:, channel, :].clone()
            if to_charges is not None:
                source = to_charges(source)
            field = descriptor.forward_dynamic(cache=cache, source_feats=source)
            if from_field is not None:
                field = from_field(field)
            # The public domain is zero field. The upstream external-field
            # contribution is exactly zero, including its coordinate derivative.
            fields.append(field / model.field_feature_norms)
        potential = torch.cat(fields, dim=-1)
        output = update(
            node_attrs=node_attrs,
            node_feats=mixed,
            edge_attrs=edge_attrs[:, : model.from_ell_max_field_update],
            edge_feats=edge_feats,
            edge_index=edge_index,
            potential_features=potential,
            local_charges=density.reshape(count, -1),
        )
        fukui = _normalized_fukui(output[:, -2:])
        density = _equilibrate(
            density + output[:, :-2].reshape(count, 2, -1), fukui, targets
        )
        histories.append(density)
        fukui_histories.append(fukui)
        potential_histories.append(potential)

    local_electron = model.local_electron_energy(
        node_attrs=node_attrs,
        node_feats=features[-1],
        edge_attrs=edge_attrs[:, : model.from_ell_max_field_update],
        edge_feats=edge_feats,
        edge_index=edge_index,
        field_feats=potential,
        charges_0=initial_density.reshape(count, -1),
        charges_induced=density.reshape(count, -1),
    ).sum()
    electron_energy = (
        local_electron if model.add_local_electron_energy else local_electron * 0
    )
    charge_density = density.sum(dim=1)
    spin_density = density[:, 0, :] - density[:, 1, :]
    if to_charges is not None:
        charge_density = to_charges(charge_density)
        spin_density = to_charges(spin_density)
        density_out = torch.stack(
            [to_charges(density[:, channel, :]) for channel in range(2)], dim=1
        )
    else:
        density_out = density
    total_charge, dipole = compute_total_charge_dipole_permuted(
        charge_density, positions, batch, 1
    )
    electrostatic = model.coulomb_energy(
        k_vectors=k_vectors,
        k_norm2=k_norm2,
        k_vector_batch=k_batch,
        k0_mask=k0_mask,
        source_feats=charge_density,
        node_positions=positions,
        batch=batch,
        volume=volume,
        pbc=pbc,
    ).sum()
    return {
        "energy": local_energy + electron_energy + electrostatic,
        "local_energy": local_energy,
        "electron_energy": electron_energy,
        "electrostatic_energy": electrostatic,
        "charges": charge_density[:, 0],
        "spins": spin_density[:, 0],
        "density_coefficients": charge_density,
        "spin_charge_density": density_out,
        "dipole": dipole[0],
        "total_charge": total_charge,
        "density_history": histories,
        "fukui_history": fukui_histories,
        "potential_history": potential_histories,
        "k_vectors_count": k_vectors.shape[0],
    }


class SymmetrixPolar(Calculator):
    """FP64 single-system Polar S energy/forces for full three-dimensional PBC.

    ``model`` is a trusted whole-model checkpoint or a PolarMACE module. The
    module is copied before conversion. Charge and spin multiplicity are read
    from ``atoms.info['charge']`` and ``atoms.info['spin']``; constructor values
    are defaults when these keys are absent. Only a zero external field is
    currently supported. Stress, FP32, other boundary conditions, training and
    multi-GPU inference are intentionally rejected.

    ``bridge=None`` (or ``'auto'``) selects DLPack on CUDA and host copies on
    CPU. ``bridge='host'`` copies features through host memory for debugging;
    ``'dlpack'`` uses device snapshots and device-to-device adjoint copies;
    it is not a zero-copy bridge. ``profile=True`` measures synchronized stage
    times for a diagnostic call. Keep profiling disabled for throughput runs.
    """

    implemented_properties = [
        "energy", "free_energy", "forces", "charges", "dipole",
        "density_coefficients", "spin_charge_density", "spins",
        "electrostatic_energy", "electron_energy",
    ]

    def __init__(
        self, model, *, device="cpu", backend=None, bridge=None,
        precision="float64", charge=None, spin=None, head=None,
        diagnostics=True, profile=False, **kwargs,
    ):
        super().__init__(**kwargs)
        if precision != "float64":
            raise ValueError("SymmetrixPolar currently supports float64 only")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("SymmetrixPolar supports CPU or a single CUDA device")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but PyTorch cannot access CUDA")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
        if bridge is None or bridge == "auto":
            bridge = "dlpack" if self.device.type == "cuda" else "host"
        if bridge not in ("host", "dlpack"):
            raise ValueError("Polar bridge must be 'auto', 'host', or 'dlpack'")
        self.bridge = bridge
        self.default_charge = 0.0 if charge is None else float(charge)
        self.default_spin = 1.0 if spin is None else float(spin)
        self.diagnostics_enabled = bool(diagnostics)
        self.profile = bool(profile)
        self._last_polar_state = None
        if isinstance(model, torch.nn.Module):
            self.checkpoint_sha256 = None
            loaded = copy.deepcopy(model).cpu()
        else:
            path = Path(model).expanduser().resolve(strict=True)
            self.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded = torch.load(path, map_location="cpu", weights_only=False)
        _validate_model(loaded)
        heads = getattr(loaded, "heads", ["Default"])
        if head is not None and head != heads[0]:
            raise ValueError(f"Unknown Polar head {head!r}; available head is {heads[0]!r}")
        loaded = loaded.to(dtype=torch.float64).eval()
        self.checkpoint_compat = repair_polar_checkpoint(loaded, pbc_handling="pbc")
        from .extract_mace_nonlinear import extract_mace_nonlinear_data

        local_data = extract_mace_nonlinear_data(loaded)
        self._artifact_directory = tempfile.TemporaryDirectory(prefix="symmetrix-polar-")
        artifact = Path(self._artifact_directory.name) / "local.json"
        artifact.write_text(json.dumps(local_data), encoding="utf-8")
        self.local_artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
        requested_backend = "cpu" if backend is None and self.device.type == "cpu" else backend
        native = load_backend(requested_backend)
        self.native_execution_space = native._kokkos_default_execution_space()
        if self.device.type == "cuda" and self.native_execution_space != "Cuda":
            raise RuntimeError(
                "CUDA Polar inference requires a CUDA Symmetrix backend; "
                f"the selected native execution space is {self.native_execution_space!r}"
            )
        if self.device.type == "cpu" and self.native_execution_space not in (
            "Serial", "OpenMP", "Threads"
        ):
            raise RuntimeError(
                "CPU Polar inference requires a host Symmetrix backend; "
                f"the selected native execution space is {self.native_execution_space!r}"
            )
        if not native._kokkos_is_initialized():
            native._init_kokkos()
        self.evaluator = native.MACENonlinearKokkos(str(artifact), heads[0])
        if not callable(getattr(self.evaluator, "_forward_features", None)):
            raise RuntimeError("The native backend lacks the Polar feature/VJP interface")
        self.native_build_info = (
            native._backend_build_info() if hasattr(native, "_backend_build_info") else {}
        )
        self.native_device_ordinal = None
        if self.device.type == "cuda":
            environment = self.evaluator.device_cuda_environment
            if environment.get("available") is not True:
                raise RuntimeError("The native Polar CUDA device is unavailable")
            self.native_device_ordinal = int(environment["device_ordinal"])
            if self.native_device_ordinal != self.device.index:
                raise RuntimeError(
                    "Native and PyTorch CUDA device ordinals differ: "
                    f"{self.native_device_ordinal} != {self.device.index}"
                )
        self.model = loaded.to(device=self.device, dtype=torch.float64).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.atomic_numbers = [int(value) for value in self.model.atomic_numbers.tolist()]
        self.cutoff = float(self.model.r_max)

    def _polar_state(self, atoms):
        charge = float(atoms.info.get("charge", self.default_charge))
        spin = float(atoms.info.get("spin", self.default_spin))
        field = np.asarray(atoms.info.get("external_field", [0.0, 0.0, 0.0]), dtype=float)
        if not math.isfinite(charge) or not math.isfinite(spin) or spin < 1 or spin != int(spin):
            raise ValueError("Polar requires finite charge and a positive integer spin multiplicity")
        if field.shape != (3,) or not np.isfinite(field).all() or np.any(field != 0):
            raise ValueError("SymmetrixPolar currently supports zero external field only")
        if not np.all(atoms.pbc) or abs(np.linalg.det(atoms.cell.array)) <= 1e-12:
            raise ValueError("SymmetrixPolar requires a nondegenerate full 3D PBC cell")
        return charge, spin

    def check_state(self, atoms, tol=1e-15):
        changes = super().check_state(atoms, tol=tol)
        if self._last_polar_state != self._polar_state(atoms):
            changes.append("polar_state")
        return changes

    def _profile_stamp(self):
        self.evaluator.fence()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return perf_counter()

    @torch.enable_grad()
    def calculate(self, atoms=None, properties=None, system_changes=None):
        timings = {}
        started = self._profile_stamp() if self.profile else 0.0
        boundary = started

        def mark(stage):
            nonlocal boundary
            if self.profile:
                now = self._profile_stamp()
                timings[stage] = (now - boundary) * 1000
                boundary = now

        properties = ["energy", "forces"] if properties is None else properties
        unsupported = set(properties) - set(self.implemented_properties)
        if unsupported:
            raise PropertyNotImplementedError(f"Unsupported Polar properties: {sorted(unsupported)}")
        super().calculate(atoms, properties, system_changes)
        # A failed recalculation must not leave values from a previous geometry
        # beside ASE's newly copied atoms and make them look cached and valid.
        self.results = {}
        atoms = self.atoms
        charge, spin = self._polar_state(atoms)
        if not len(atoms):
            raise ValueError("Polar requires at least one atom")
        unsupported_species = sorted(set(atoms.numbers) - set(self.atomic_numbers))
        if unsupported_species:
            raise ValueError(f"Polar checkpoint does not support atomic numbers {unsupported_species}")
        receiver, sender, shifts = neighbor_list("ijS", atoms, self.cutoff)
        order = np.argsort(receiver, kind="stable")
        receiver, sender, shifts = receiver[order], sender[order], shifts[order]
        # ASE D points receiver -> sender; MACE SH uses sender -> receiver.
        vectors_np = atoms.positions[receiver] - atoms.positions[sender] - shifts @ atoms.cell.array
        lengths_np = np.linalg.norm(vectors_np, axis=1)
        if np.any(lengths_np <= 0) or not np.isfinite(vectors_np).all():
            raise ValueError("Polar graph contains invalid edge vectors")
        type_lookup = {number: index for index, number in enumerate(self.atomic_numbers)}
        node_types = np.asarray([type_lookup[int(n)] for n in atoms.numbers], dtype=np.int32)
        native_vectors = np.ascontiguousarray(vectors_np[:, [1, 2, 0]]).reshape(-1)
        positions = torch.tensor(atoms.positions, device=self.device, dtype=torch.float64, requires_grad=True)
        cell = torch.tensor(atoms.cell.array, device=self.device, dtype=torch.float64)
        edge_index = torch.tensor(np.stack([sender, receiver]), device=self.device, dtype=torch.long)
        shift_tensor = torch.tensor(shifts, device=self.device, dtype=torch.float64)
        vectors = positions[edge_index[1]] - positions[edge_index[0]] - shift_tensor @ cell
        node_attrs = torch.nn.functional.one_hot(
            torch.tensor(node_types, device=self.device, dtype=torch.long), len(self.atomic_numbers)
        ).to(torch.float64)
        mark("graph")
        token = self.evaluator._forward_features(
            len(atoms), node_types, np.bincount(receiver, minlength=len(atoms)).astype(np.int32),
            sender.astype(np.int32), node_types[sender], native_vectors,
            np.ascontiguousarray(lengths_np),
        )
        mark("native_forward")
        if self.bridge == "dlpack":
            features = [torch.utils.dlpack.from_dlpack(value) for value in self.evaluator._feature_tensors(token)]
            if any(value.device != self.device for value in features):
                raise RuntimeError("Native and PyTorch Polar devices differ")
        else:
            features = [torch.tensor(value, device=self.device, dtype=torch.float64) for value in self.evaluator._feature_arrays(token)]
        features = [value.detach().requires_grad_(True) for value in features]
        mark("feature_export")
        with torch.enable_grad():
            output = _suffix(self.model, features, positions, cell, node_attrs, edge_index, vectors, charge, spin)
            mark("torch_suffix_forward")
            gradients = torch.autograd.grad(output["energy"], [*features, positions], allow_unused=True)
        feature_gradients = [torch.zeros_like(value) if grad is None else grad.contiguous() for value, grad in zip(features, gradients[:-1])]
        direct_force = torch.zeros_like(positions) if gradients[-1] is None else -gradients[-1]
        mark("torch_autograd")
        if self.bridge == "dlpack":
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.evaluator._reverse_features(token, [torch.utils.dlpack.to_dlpack(value) for value in feature_gradients])
            mark("native_vjp")
            edge_force = torch.utils.dlpack.from_dlpack(self.evaluator._edge_force_tensor())
            edge_force = edge_force[:, [2, 0, 1]]
        else:
            self.evaluator._reverse_features_host(token, [value.detach().cpu().numpy() for value in feature_gradients])
            mark("native_vjp")
            edge_force = torch.tensor(np.asarray(self.evaluator.node_forces).reshape(-1, 3)[:, [2, 0, 1]], device=self.device, dtype=torch.float64)
        forces = direct_force.clone()
        forces.index_add_(0, edge_index[1], edge_force)
        forces.index_add_(0, edge_index[0], -edge_force)
        if not torch.isfinite(output["energy"]).all() or not torch.isfinite(forces).all():
            raise FloatingPointError("Non-finite complete Polar energy or force")
        self.results = {
            "energy": float(output["energy"].detach().cpu()),
            "free_energy": float(output["energy"].detach().cpu()),
            "forces": forces.detach().cpu().numpy().copy(),
        }
        for name in ("charges", "spins", "density_coefficients", "spin_charge_density", "dipole"):
            self.results[name] = output[name].detach().cpu().numpy().copy()
        for name in ("electron_energy", "electrostatic_energy"):
            self.results[name] = float(output[name].detach().cpu())
        totals = [value[:, :, 0].detach().sum(dim=0).cpu().tolist() for value in output["density_history"]]
        self.results["polar_diagnostics"] = {
            "backbone": "native_feature_forward", "prefix_fallback": False,
            "bridge": self.bridge, "precision": "float64",
            "bridge_semantics": (
                "device_snapshot_and_d2d_seed_copy"
                if self.bridge == "dlpack" else "host_snapshot_copy"
            ),
            "feature_device": str(features[0].device),
            "native_class": type(self.evaluator).__module__ + "." + type(self.evaluator).__name__,
            "native_build_info": self.native_build_info,
            "native_execution_space": self.native_execution_space,
            "native_device_ordinal": self.native_device_ordinal,
            "checkpoint_sha256": self.checkpoint_sha256,
            "local_artifact_sha256": self.local_artifact_sha256,
            "checkpoint_compat": self.checkpoint_compat,
            "num_interactions": len(features), "num_recursion_steps": len(output["potential_history"]),
            "feature_shapes": [list(value.shape) for value in features],
            "spherical_vector_permutation": [1, 2, 0],
            "electrostatic_boundary": "pbc_k0_removed", "k_vectors_count": output["k_vectors_count"],
            "normalized_spin_charge_totals": totals,
            "total_charge": float(output["total_charge"].detach().sum().cpu()),
            "total_spin": float(output["spins"].detach().sum().cpu()),
            "energy_terms": {name: float(output[key].detach().cpu()) for name, key in (
                ("local", "local_energy"), ("electron", "electron_energy"),
                ("electrostatic", "electrostatic_energy"), ("total", "energy"))},
        }
        if self.diagnostics_enabled:
            self.results["polar_diagnostics"]["density_history"] = [value.detach().cpu().numpy().copy() for value in output["density_history"]]
            self.results["polar_diagnostics"]["fukui_history"] = [value.detach().cpu().numpy().copy() for value in output["fukui_history"]]
        self._last_polar_state = (charge, spin)
        mark("force_result_assembly")
        if self.profile:
            timings["total"] = (boundary - started) * 1000
            self.results["polar_diagnostics"]["timings_ms"] = timings
            self.results["polar_diagnostics"]["timing_policy"] = (
                "synchronized_diagnostic_call_excluded_from_throughput"
            )
