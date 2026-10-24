#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime_api.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

#include "device_environment.hpp"

#include "mace_nonlinear_kokkos.hpp"
#include "utilities_kokkos.hpp"

namespace py=pybind11;
using IntArray=py::array_t<int,py::array::c_style|py::array::forcecast>;
using Int64Array=py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>;
using DoubleArray=py::array_t<double,py::array::c_style|py::array::forcecast>;

namespace {
// DLPack's stable legacy ABI. Keeping these declarations local avoids a
// dependency on either LibTorch or an optional Python DLPack header package.
struct FeatureDLDevice { std::int32_t type,id; };
struct FeatureDLDataType { std::uint8_t code,bits; std::uint16_t lanes; };
struct FeatureDLTensor {
    void* data;
    FeatureDLDevice device;
    std::int32_t ndim;
    FeatureDLDataType dtype;
    std::int64_t* shape;
    std::int64_t* strides;
    std::uint64_t byte_offset;
};
struct FeatureDLManagedTensor {
    FeatureDLTensor tensor;
    void* manager_ctx;
    void (*deleter)(FeatureDLManagedTensor*);
};

FeatureDLDevice feature_dl_device()
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>) {
        int ordinal=0;
        if(cudaGetDevice(&ordinal)!=cudaSuccess)
            throw std::runtime_error("Could not query the feature CUDA device.");
        return {2,ordinal};
    }
#elif defined(KOKKOS_ENABLE_HIP)
    if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>) {
        int ordinal=0;
        if(hipGetDevice(&ordinal)!=hipSuccess)
            throw std::runtime_error("Could not query the feature HIP device.");
        return {10,ordinal};
    }
#endif
    return {1,0};
}

void synchronize_feature_producer()
{
    // Legacy DLPack capsules do not carry a producer stream. Synchronizing
    // before the device copy establishes ownership without host staging.
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>)
        if(cudaDeviceSynchronize()!=cudaSuccess)
            throw std::runtime_error("Could not synchronize feature CUDA seeds.");
#elif defined(KOKKOS_ENABLE_HIP)
    if constexpr(device_execution_space<Kokkos::DefaultExecutionSpace>)
        if(hipDeviceSynchronize()!=hipSuccess)
            throw std::runtime_error("Could not synchronize feature HIP seeds.");
#endif
}

template<typename Precision>
struct ExportedFeatureTensor {
    KokkosLiveObjectGuard live_guard;
    Kokkos::View<Precision**,Kokkos::LayoutRight> values;
    std::array<std::int64_t,2> shape,strides;
    FeatureDLManagedTensor managed{};
};

template<typename View>
py::capsule export_feature_tensor(const View& source)
{
    using Precision=typename View::non_const_value_type;
    using Owner=ExportedFeatureTensor<Precision>;
    auto owner=std::make_unique<Owner>();
    owner->values=decltype(owner->values)(
        "exported nonlinear feature",source.extent(0),source.extent(1));
    Kokkos::deep_copy(owner->values,source);
    Kokkos::fence("Export nonlinear DLPack snapshot");
    owner->shape={static_cast<std::int64_t>(source.extent(0)),
        static_cast<std::int64_t>(source.extent(1))};
    owner->strides={static_cast<std::int64_t>(owner->values.stride(0)),1};
    owner->managed.tensor={owner->values.data(),feature_dl_device(),2,
        {2,static_cast<std::uint8_t>(8*sizeof(Precision)),1},
        owner->shape.data(),owner->strides.data(),0};
    owner->managed.manager_ctx=owner.get();
    owner->managed.deleter=[](FeatureDLManagedTensor* managed) {
        delete static_cast<Owner*>(managed->manager_ctx);
    };
    auto* managed=&owner->managed;
    py::capsule capsule(managed,"dltensor",[](PyObject* object) {
        if(PyCapsule_IsValid(object,"dltensor")) {
            auto* value=static_cast<FeatureDLManagedTensor*>(
                PyCapsule_GetPointer(object,"dltensor"));
            if(value&&value->deleter)value->deleter(value);
        }
    });
    owner.release();
    return capsule;
}

template<typename Precision>
Kokkos::View<Precision**,Kokkos::LayoutRight> import_feature_tensor(
    const py::handle capsule)
{
    if(!PyCapsule_IsValid(capsule.ptr(),"dltensor"))
        throw py::value_error("Feature seeds require unused DLPack capsules.");
    auto* managed=static_cast<FeatureDLManagedTensor*>(
        PyCapsule_GetPointer(capsule.ptr(),"dltensor"));
    const auto& tensor=managed->tensor;
    const auto device=feature_dl_device();
    if(tensor.device.type!=device.type||tensor.device.id!=device.id)
        throw py::value_error("Feature seed and native evaluator devices differ.");
    if(tensor.ndim!=2||!tensor.shape||tensor.dtype.code!=2
        ||tensor.dtype.bits!=8*sizeof(Precision)||tensor.dtype.lanes!=1)
        throw py::value_error("Feature seeds require rank-two native-precision floats.");
    for(int axis=0;axis<2;++axis)
        if(tensor.shape[axis]<0
            ||tensor.shape[axis]>std::numeric_limits<int>::max())
            throw py::value_error("Feature seed extent is out of range.");
    if(tensor.strides&&(tensor.strides[1]!=1
        ||tensor.strides[0]!=tensor.shape[1]))
        throw py::value_error("Feature seeds must be contiguous.");
    if(tensor.byte_offset%alignof(Precision)!=0
        ||(!tensor.data&&tensor.shape[0]!=0&&tensor.shape[1]!=0))
        throw py::value_error("Feature seed storage is invalid.");
    using Unmanaged=Kokkos::View<const Precision**,Kokkos::LayoutRight,
        typename Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::MemoryUnmanaged>;
    const auto* pointer=tensor.data?reinterpret_cast<const Precision*>(
        static_cast<const unsigned char*>(tensor.data)+tensor.byte_offset):nullptr;
    Unmanaged source(pointer,tensor.shape[0],tensor.shape[1]);
    Kokkos::View<Precision**,Kokkos::LayoutRight> result(
        "imported feature seed",tensor.shape[0],tensor.shape[1]);
    Kokkos::deep_copy(result,source);
    Kokkos::fence("Import nonlinear DLPack seed");
    if(PyCapsule_SetName(capsule.ptr(),"used_dltensor")!=0)
        throw py::error_already_set();
    if(managed->deleter)managed->deleter(managed);
    return result;
}
} // namespace

template<typename Evaluator>
void bind_mace_nonlinear_kokkos_evaluator(
    py::module_& module,const char* name)
{
    py::class_<Evaluator>(module,name)
        .def(py::init<const std::string&, const std::string&>(),
            py::arg("filename"), py::arg("head") = "")
        .def_readonly("r_cut",&Evaluator::r_cut)
        .def_readonly("has_field_coupling",&Evaluator::has_field_coupling)
        .def_readonly("selected_head",&Evaluator::selected_head)
        .def_readonly("available_heads",&Evaluator::available_heads)
        .def_property_readonly("is_mh1_family",&Evaluator::is_mh1_family)
        .def_property_readonly("mh1_node_channels",&Evaluator::mh1_node_channels)
        .def_property_readonly("mh1_edge_channels",&Evaluator::mh1_edge_channels)
        .def_property_readonly("mh1_radial_size",&Evaluator::mh1_radial_size)
        .def_property_readonly("mh1_l_max",&Evaluator::mh1_l_max)
        .def_property_readonly(
            "mh1_family_rejection_reason",
            &Evaluator::mh1_family_rejection_reason)
        .def_property_readonly(
            "mh1_uses_compiled_products",
            &Evaluator::mh1_uses_compiled_products)
        .def_property_readonly(
            "mh1_uses_pair_conditioning",
            &Evaluator::mh1_uses_pair_conditioning)
        .def_property_readonly(
            "mh1_uses_external_uvu_tensors",
            &Evaluator::mh1_uses_external_uvu_tensors)
        .def_property_readonly(
            "mh1_fast_path_rejection_reason",
            &Evaluator::mh1_fast_path_rejection_reason)
        .def_property_readonly("scalar_size_bytes",[](const Evaluator&) {
            return sizeof(typename Evaluator::precision_type);
        })
        .def_property_readonly("uses_mh1_fast_path",&Evaluator::uses_mh1_fast_path)
        .def_property_readonly("supports_streamed_edges",&Evaluator::supports_streamed_edges)
        .def_property_readonly("supports_factorized",&Evaluator::supports_factorized)
        .def_property_readonly(
            "has_execution_mh1_contract",&Evaluator::has_execution_mh1_contract)
        .def_property_readonly(
            "execution_mh1_generation_fingerprint",
            &Evaluator::execution_mh1_generation_fingerprint)
        .def_property_readonly(
            "execution_mh1_semantic_fingerprint",
            &Evaluator::execution_mh1_semantic_fingerprint)
        .def_property_readonly(
            "execution_mh1_structure_fingerprint",
            &Evaluator::execution_mh1_structure_fingerprint)
        .def(
            "_set_execution_mh1_contract_identity",
            &Evaluator::set_execution_mh1_contract_identity)
        .def(
            "_set_execution_mh1_v4_contract_identity",
            &Evaluator::set_execution_mh1_v4_contract_identity)
        .def("_load_jit_mh1_host_plugin",
            &Evaluator::load_jit_mh1_host_plugin)
        .def("_load_jit_mh1_host_plugin_v4",
            &Evaluator::load_jit_mh1_host_plugin_v4)
        .def("_load_jit_mh1_host_plugin_v5",
            &Evaluator::load_jit_mh1_host_plugin_v5)
        .def_property_readonly(
            "jit_mh1_host_plugin_ready",
            &Evaluator::jit_mh1_host_plugin_ready)
        .def_property_readonly(
            "jit_mh1_host_plugin_v4_ready",
            &Evaluator::jit_mh1_host_plugin_v4_ready)
        .def_property_readonly(
            "jit_mh1_host_plugin_v5_ready",
            &Evaluator::jit_mh1_host_plugin_v5_ready)
        .def_property_readonly(
            "jit_mh1_host_plugin_path",
            &Evaluator::jit_mh1_host_plugin_path)
        .def_property_readonly(
            "jit_mh1_host_plugin_artifact_id",
            &Evaluator::jit_mh1_host_plugin_artifact_id)
        .def_property_readonly(
            "execution_mh1_retained_interaction_output_dimensions",
            &Evaluator::execution_mh1_retained_interaction_output_dimensions)
        .def("_load_jit_mh1_cuda_plugin",
            &Evaluator::load_jit_mh1_cuda_plugin)
        .def("_load_jit_mh1_cuda_plugin_v4",
            &Evaluator::load_jit_mh1_cuda_plugin_v4,
            py::arg("path"),py::arg("launch_plan_json")="")
        .def_property_readonly(
            "jit_mh1_cuda_plugin_ready",
            &Evaluator::jit_mh1_cuda_plugin_ready)
        .def_property_readonly(
            "jit_mh1_cuda_plugin_v4_ready",
            &Evaluator::jit_mh1_cuda_plugin_v4_ready)
        .def_property_readonly(
            "jit_mh1_cuda_plugin_path",
            &Evaluator::jit_mh1_cuda_plugin_path)
        .def_property_readonly(
            "jit_mh1_cuda_plugin_artifact_id",
            &Evaluator::jit_mh1_cuda_plugin_artifact_id)
        .def_property_readonly(
            "device_cuda_environment",
            [] (const Evaluator& self) {
                py::dict result;
                result["available"] = self.device_cuda_available();
                if (!self.device_cuda_available())
                    return result;
                const int compute_capability =
                    self.execution_cuda_compute_capability();
                result["backend"] = "cuda";
                result["device_name"] = self.execution_cuda_device_name();
                result["device_ordinal"] = self.execution_cuda_device_ordinal();
                result["compute_capability"] =
                    std::to_string(compute_capability/10)+"."
                    +std::to_string(compute_capability%10);
                result["compute_capability_code"] = compute_capability;
                result["multiprocessor_count"] =
                    self.execution_cuda_multiprocessor_count();
                result["warp_width"] = self.execution_cuda_warp_width();
                result["runtime_version"] = self.execution_cuda_runtime_version();
                result["driver_version"] = self.execution_cuda_driver_version();
                return result;
            })
        .def_property_readonly(
            "execution_device_execution_environment",
            [] (const Evaluator&) { return execution_device_environment_dict(); })
        .def_property_readonly(
            "execution_mh1_execution_backend",
            &Evaluator::execution_mh1_execution_backend)
        .def_property_readonly("streamed_edges_mode",&Evaluator::streamed_edges_mode)
        .def("set_streamed_edges",&Evaluator::set_streamed_edges)
        .def_property_readonly("mh1_edge_executor",&Evaluator::mh1_edge_executor)
        .def("set_mh1_edge_executor",&Evaluator::set_mh1_edge_executor)
        .def_property_readonly(
            "mh1_pair_spline_ready",&Evaluator::mh1_pair_spline_ready)
        .def_property_readonly(
            "mh1_pair_spline_nodes",&Evaluator::mh1_pair_spline_nodes)
        .def_property_readonly(
            "mh1_pair_spline_type_count",&Evaluator::mh1_pair_spline_type_count)
        .def(
            "_set_execution_mh1_node_arena_policy",
            &Evaluator::set_execution_mh1_node_arena_policy)
        .def_property_readonly(
            "execution_mh1_node_arena_policy",
            &Evaluator::execution_mh1_node_arena_policy)
        .def_property_readonly(
            "execution_mh1_node_arena_tile_rows",
            &Evaluator::execution_mh1_node_arena_tile_rows)
        .def(
            "_set_execution_mh1_node_arena_tile_rows_for_testing",
            &Evaluator::set_execution_mh1_node_arena_tile_rows_for_testing)
        .def_property_readonly(
            "factorized_forward_evaluation_count",
            &Evaluator::factorized_forward_evaluation_count)
        .def_property_readonly(
            "factorized_reverse_evaluation_count",
            &Evaluator::factorized_reverse_evaluation_count)
        .def_property_readonly(
            "factorized_fallback_evaluation_count",
            &Evaluator::factorized_fallback_evaluation_count)
        .def_property_readonly(
            "factorized_schedule_build_count",
            &Evaluator::factorized_schedule_build_count)
        .def_property_readonly(
            "factorized_schedule_entries",
            &Evaluator::factorized_schedule_entries)
        .def_property_readonly(
            "factorized_schedule_active_sources",
            &Evaluator::factorized_schedule_active_sources)
        .def_property_readonly(
            "factorized_schedule_bytes",
            &Evaluator::factorized_schedule_bytes)
        .def_property_readonly(
            "factorized_graph_generation",
            &Evaluator::factorized_graph_generation)
        .def_property_readonly(
            "factorized_prepared_graph_count",
            &Evaluator::factorized_prepared_graph_count)
        .def_property_readonly(
            "factorized_prepared_evaluation_count",
            &Evaluator::factorized_prepared_evaluation_count)
        .def_property_readonly(
            "factorized_topology_validation_count",
            &Evaluator::factorized_topology_validation_count)
        .def_property_readonly(
            "factorized_topology_validation_skip_count",
            &Evaluator::factorized_topology_validation_skip_count)
        .def_property_readonly(
            "factorized_preparation_fence_count",
            &Evaluator::factorized_preparation_fence_count)
        .def_property_readonly(
            "factorized_evaluation_fence_count",
            &Evaluator::factorized_evaluation_fence_count)
        .def_property_readonly(
            "factorized_stage_fence_count",
            &Evaluator::factorized_stage_fence_count)
        .def_property_readonly(
            "factorized_zbl_evaluator_stream_launch_count",
            &Evaluator::factorized_zbl_evaluator_stream_launch_count)
        .def_property_readonly(
            "execution_geometry_capacity_edges",
            &Evaluator::execution_geometry_capacity_edges)
        .def_property_readonly(
            "execution_geometry_growth_reason",
            &Evaluator::execution_geometry_growth_reason)
        .def(
            "_set_execution_geometry_device_memory_info_for_testing",
            &Evaluator::set_execution_geometry_device_memory_info_for_testing)
        .def_property_readonly(
            "execution_geometry_workspace_bytes",
            &Evaluator::execution_geometry_workspace_bytes)
        .def(
            "_set_execution_mh1_scratch_budget_bytes",
            [] (Evaluator& self,const py::ssize_t budget_bytes) {
                if(budget_bytes<0)
                    throw py::value_error(
                        "Execution MH-1 scratch budget must be non-negative.");
                self.set_execution_mh1_scratch_budget_bytes(
                    static_cast<std::size_t>(budget_bytes));
            },
            py::arg("budget_bytes"))
        .def_property_readonly(
            "execution_mh1_scratch_budget_bytes",
            &Evaluator::execution_mh1_scratch_budget_bytes)
        .def_property_readonly(
            "execution_mh1_scratch_minimum_bytes",
            &Evaluator::execution_mh1_scratch_minimum_bytes)
        .def_property_readonly(
            "execution_mh1_scratch_planned_bytes",
            &Evaluator::execution_mh1_scratch_planned_bytes)
        .def_property_readonly(
            "execution_mh1_scratch_retained_bytes",
            &Evaluator::execution_mh1_scratch_retained_bytes)
        .def_property_readonly(
            "execution_mh1_scratch_recomputed_bytes",
            &Evaluator::execution_mh1_scratch_recomputed_bytes)
        .def_property_readonly(
            "execution_mh1_scratch_recomputed_layer_count",
            &Evaluator::execution_mh1_scratch_recomputed_layer_count)
        .def_property_readonly(
            "execution_mh1_scratch_budget_satisfied",
            &Evaluator::execution_mh1_scratch_budget_satisfied)
        .def_property_readonly(
            "execution_mh1_conditioning_recomputation_count",
            &Evaluator::execution_mh1_conditioning_recomputation_count)
        .def_property_readonly(
            "execution_geometry_allocation_count",
            &Evaluator::execution_geometry_allocation_count)
        .def_property_readonly(
            "execution_geometry_copy_count",
            &Evaluator::execution_geometry_copy_count)
        .def_property_readonly(
            "execution_sphericart_initialization_count",
            &Evaluator::execution_sphericart_initialization_count)
        .def_property_readonly(
            "execution_sphericart_launch_count",
            &Evaluator::execution_sphericart_launch_count)
        .def_property_readonly(
            "execution_sphericart_async_launch_count",
            &Evaluator::execution_sphericart_async_launch_count)
        .def_property_readonly(
            "execution_mh1_generated_forward_launch_count",
            &Evaluator::execution_mh1_generated_forward_launch_count)
        .def_property_readonly(
            "execution_mh1_generated_source_reverse_launch_count",
            &Evaluator::execution_mh1_generated_source_reverse_launch_count)
        .def_property_readonly(
            "execution_mh1_generated_edge_reverse_launch_count",
            &Evaluator::execution_mh1_generated_edge_reverse_launch_count)
        .def_property_readonly(
            "execution_mh1_generated_conditioning_forward_launch_count",
            &Evaluator::execution_mh1_generated_conditioning_forward_launch_count)
        .def_property_readonly(
            "execution_mh1_generated_conditioning_reverse_launch_count",
            &Evaluator::execution_mh1_generated_conditioning_reverse_launch_count)
        .def_property_readonly(
            "factorized_source_owned_reverse",
            &Evaluator::factorized_source_owned_reverse)
        .def_property_readonly("edge_workspace_rows",&Evaluator::edge_workspace_rows)
        .def_property_readonly("edge_workspace_bytes",&Evaluator::edge_workspace_bytes)
        .def_property_readonly(
            "conditioned_mlp_workspace_bytes",
            &Evaluator::conditioned_mlp_workspace_bytes)
        .def("fence",&Evaluator::fence)
        .def("set_e3_linear_backend",&Evaluator::set_e3_linear_backend)
        .def_property_readonly("e3_linear_backend",&Evaluator::e3_linear_backend)
        .def("selected_e3_linear_backend",&Evaluator::selected_e3_linear_backend)
        .def(
            "set_execution_mh1_host_node_backend",
            &Evaluator::set_execution_mh1_host_node_backend)
        .def_property_readonly(
            "execution_mh1_host_node_backend",
            &Evaluator::execution_mh1_host_node_backend)
        .def_property_readonly(
            "selected_execution_mh1_host_node_backend",
            &Evaluator::selected_execution_mh1_host_node_backend)
        .def_property_readonly("tensor_product_backend",&Evaluator::tensor_product_backend)
        .def_property_readonly(
            "tensor_product_execution_backend",
            &Evaluator::tensor_product_execution_backend)
        .def_property_readonly(
            "tensor_product_channel_team_size",
            &Evaluator::tensor_product_channel_team_size)
        .def_property_readonly(
            "tensor_product_harmonic_team_size",
            &Evaluator::tensor_product_harmonic_team_size)
        .def(
            "set_fused_gate_normalization_reverse",
            &Evaluator::set_fused_gate_normalization_reverse)
        .def_property_readonly(
            "fused_gate_normalization_reverse_available",
            &Evaluator::fused_gate_normalization_reverse_available)
        .def_property_readonly(
            "uses_fused_gate_normalization_reverse",
            &Evaluator::uses_fused_gate_normalization_reverse)
        .def(
            "set_direct_node_tensor_reverse",
            &Evaluator::set_direct_node_tensor_reverse)
        .def_property_readonly(
            "direct_node_tensor_reverse_available",
            &Evaluator::direct_node_tensor_reverse_available)
        .def_property_readonly(
            "uses_direct_node_tensor_reverse",
            &Evaluator::uses_direct_node_tensor_reverse)
        .def_property_readonly("linear_workspace_bytes",&Evaluator::linear_workspace_bytes)
        .def_property_readonly("tensor_workspace_bytes",&Evaluator::tensor_workspace_bytes)
        .def_property_readonly("product_workspace_bytes",&Evaluator::product_workspace_bytes)
        .def_property_readonly("node_workspace_bytes",&Evaluator::node_workspace_bytes)
        .def_property_readonly("precision_workspace_bytes",&Evaluator::precision_workspace_bytes)
        .def_property_readonly("atomic_numbers",[](Evaluator& self){return self.atomic_numbers_host;})
        .def_property_readonly("atomic_energies",[](Evaluator& self){return view2vector(self.atomic_energies);})
        .def_property_readonly("node_energies",[](Evaluator& self){return view2vector(self.node_energies);})
        .def_property_readonly("node_forces",[](Evaluator& self){return view2vector(self.node_forces);})
        .def("_prepare_factorized_graph",
            [] (Evaluator& self,const int num_nodes,IntArray node_types,
                    IntArray num_neigh,IntArray neigh_indices,
                    IntArray neigh_types) {
                return self.prepare_factorized_graph(
                    num_nodes,
                    std::span<const int>(node_types.data(),node_types.size()),
                    std::span<const int>(num_neigh.data(),num_neigh.size()),
                    std::span<const int>(
                        neigh_indices.data(),neigh_indices.size()),
                    std::span<const int>(neigh_types.data(),neigh_types.size()));
            },
            py::arg("num_nodes"),py::arg("node_types"),
            py::arg("num_neigh"),py::arg("neigh_indices"),
            py::arg("neigh_types"))
        .def("_prepare_factorized_geometry",
            [] (Evaluator& self,const std::uint64_t graph_generation,
                    DoubleArray reference_positions,DoubleArray reference_xyz,
                    DoubleArray cell,DoubleArray inverse_cell,IntArray pbc) {
                self.prepare_factorized_geometry(
                    graph_generation,
                    std::span<const double>(
                        reference_positions.data(),reference_positions.size()),
                    std::span<const double>(
                        reference_xyz.data(),reference_xyz.size()),
                    std::span<const double>(cell.data(),cell.size()),
                    std::span<const double>(
                        inverse_cell.data(),inverse_cell.size()),
                    std::span<const int>(pbc.data(),pbc.size()));
            },
            py::arg("graph_generation"),py::arg("reference_positions"),
            py::arg("reference_xyz"),py::arg("cell"),
            py::arg("inverse_cell"),py::arg("pbc"))
        .def("_compute_prepared_factorized",
            [] (Evaluator& self,const std::uint64_t graph_generation,
                    DoubleArray xyz,DoubleArray distances) {
                self.compute_prepared_factorized(
                    graph_generation,
                    std::span<const double>(xyz.data(),xyz.size()),
                    std::span<const double>(
                        distances.data(),distances.size()));
            },
            py::arg("graph_generation"),py::arg("xyz"),
            py::arg("distances"))
        .def("_compute_prepared_factorized_positions",
            [] (Evaluator& self,const std::uint64_t graph_generation,
                    DoubleArray positions) {
                self.compute_prepared_factorized_positions(
                    graph_generation,
                    std::span<const double>(positions.data(),positions.size()));
            },
            py::arg("graph_generation"),py::arg("positions"))
        .def("_reduce_atom_forces",
            [] (Evaluator& self,const int num_nodes,IntArray edge_receivers,
                    IntArray edge_sources,const std::uint64_t graph_generation) {
                if(num_nodes<0
                    ||edge_receivers.size()!=edge_sources.size())
                    throw py::value_error(
                        "MH-1 atom-force reduction extents are inconsistent.");
                if(graph_generation!=0)
                    self.reduce_prepared_node_forces(graph_generation);
                else
                    self.reduce_node_forces(
                        num_nodes,
                        create_kokkos_view(
                            "MH-1 force reduction receivers",edge_receivers),
                        create_kokkos_view(
                            "MH-1 force reduction sources",edge_sources));
                if(self.atom_forces.size()
                    !=3*static_cast<std::size_t>(num_nodes))
                    throw py::value_error(
                        "MH-1 atom-force reduction node extent is inconsistent.");
                return view2vector(self.atom_forces);
            },
            py::arg("num_nodes"),py::arg("edge_receivers"),
            py::arg("edge_sources"),py::arg("graph_generation")=0)
        .def("_reduce_stress",
            [] (Evaluator& self,const double volume,DoubleArray xyz,
                    const std::uint64_t graph_generation) {
                if(graph_generation!=0)
                    self.reduce_prepared_stress(volume,graph_generation);
                else
                    self.reduce_stress(
                        volume,create_kokkos_view("MH-1 stress xyz",xyz));
                if(self.stress_tensor.size()!=9)
                    throw py::value_error(
                        "MH-1 stress reduction tensor extent is inconsistent.");
                return view2vector(self.stress_tensor);
            },
            py::arg("volume"),py::arg("xyz"),
            py::arg("graph_generation")=0)
        .def("_prepare_factorized_batch",
            [] (Evaluator& self,const std::uint64_t graph_generation,
                    Int64Array edge_offsets) {
                self.prepare_factorized_batch(
                    graph_generation,
                    std::span<const std::int64_t>(
                        edge_offsets.data(),edge_offsets.size()));
            },
            py::arg("graph_generation"),py::arg("edge_offsets"))
        .def("_reduce_batched_stress",
            [] (Evaluator& self,DoubleArray volumes,
                    const std::uint64_t graph_generation) {
                self.reduce_prepared_batched_stress(
                    std::span<const double>(volumes.data(),volumes.size()),
                    graph_generation);
                if(self.stress_tensor.size()!=9*volumes.size())
                    throw py::value_error(
                        "MH-1 batched stress tensor extent is inconsistent.");
                return view2vector(self.stress_tensor);
            },
            py::arg("volumes"),py::arg("graph_generation"))
        .def("_forward_features",
            [](Evaluator& self,int num_nodes,IntArray node_types,
                    IntArray num_neigh,IntArray neigh_indices,
                    IntArray neigh_types,DoubleArray xyz,DoubleArray distances) {
                return self.forward_features(num_nodes,
                    create_kokkos_view("feature node types",node_types),
                    create_kokkos_view("feature num neigh",num_neigh),
                    create_kokkos_view("feature neigh indices",neigh_indices),
                    create_kokkos_view("feature neigh types",neigh_types),
                    create_kokkos_view("feature xyz",xyz),
                    create_kokkos_view("feature distances",distances));
            },py::arg("num_nodes"),py::arg("node_types"),py::arg("num_neigh"),
            py::arg("neigh_indices"),py::arg("neigh_types"),py::arg("xyz"),
            py::arg("distances"))
        .def("_feature_arrays",[](Evaluator& self,std::uint64_t token) {
            using Precision=typename Evaluator::precision_type;
            py::list result;
            for(const auto& view:self.feature_views(token)) {
                const auto host=Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(),view);
                py::array_t<Precision> array({
                    static_cast<py::ssize_t>(host.extent(0)),
                    static_cast<py::ssize_t>(host.extent(1))});
                auto output=array.template mutable_unchecked<2>();
                for(std::size_t i=0;i<host.extent(0);++i)
                    for(std::size_t j=0;j<host.extent(1);++j)
                        output(i,j)=host(i,j);
                result.append(std::move(array));
            }
            return result;
        },py::arg("token"))
        .def("_feature_tensors",[](Evaluator& self,std::uint64_t token) {
            py::list result;
            for(const auto& view:self.feature_views(token))
                result.append(export_feature_tensor(view));
            return result;
        },py::arg("token"))
        .def("_reverse_features_host",
            [](Evaluator& self,std::uint64_t token,py::sequence arrays) {
                using Precision=typename Evaluator::precision_type;
                using Array=py::array_t<Precision,
                    py::array::c_style|py::array::forcecast>;
                auto owned=std::vector<typename Evaluator::FeatureView>();
                auto adjoints=std::vector<typename Evaluator::ConstFeatureView>();
                for(const auto& item:arrays) {
                    auto array=Array::ensure(item);
                    if(!array||array.ndim()!=2)
                        throw py::value_error("Feature seeds require rank-two arrays.");
                    Kokkos::View<const Precision**,Kokkos::LayoutRight,
                        Kokkos::HostSpace,Kokkos::MemoryUnmanaged> host(
                            array.data(),array.shape(0),array.shape(1));
                    typename Evaluator::FeatureView value(
                        "host feature seed",array.shape(0),array.shape(1));
                    Kokkos::deep_copy(value,host);
                    owned.emplace_back(std::move(value));
                    adjoints.emplace_back(owned.back());
                }
                self.reverse_features(token,adjoints);
            },py::arg("token"),py::arg("adjoints"))
        .def("_reverse_features",
            [](Evaluator& self,std::uint64_t token,py::sequence capsules) {
                using Precision=typename Evaluator::precision_type;
                // Check the token before consuming any one-shot capsules.
                self.feature_views(token);
                synchronize_feature_producer();
                auto owned=std::vector<typename Evaluator::FeatureView>();
                auto adjoints=std::vector<typename Evaluator::ConstFeatureView>();
                for(const auto& capsule:capsules) {
                    owned.emplace_back(import_feature_tensor<Precision>(capsule));
                    adjoints.emplace_back(owned.back());
                }
                self.reverse_features(token,adjoints);
            },py::arg("token"),py::arg("adjoints"))
        .def("_edge_force_tensor",[](Evaluator& self) {
            if(self.node_forces.size()%3!=0)
                throw py::value_error("Native edge force extent is inconsistent.");
            using View=Kokkos::View<const double**,Kokkos::LayoutRight,
                typename Kokkos::DefaultExecutionSpace::memory_space,
                Kokkos::MemoryUnmanaged>;
            return export_feature_tensor(View(self.node_forces.data(),
                self.node_forces.size()/3,3));
        })
        .def("compute_node_energies_forces",
            [](Evaluator& self,int num_nodes,IntArray node_types,
                    IntArray num_neigh,IntArray neigh_indices,
                    IntArray neigh_types,DoubleArray xyz,
                    DoubleArray distances) {
                try {
                    self.compute_node_energies_forces(
                        num_nodes,
                        create_kokkos_view("nonlinear node types",node_types),
                        create_kokkos_view("nonlinear num neigh",num_neigh),
                        create_kokkos_view("nonlinear neigh indices",neigh_indices),
                        create_kokkos_view("nonlinear neigh types",neigh_types),
                        create_kokkos_view("nonlinear xyz",xyz),
                        create_kokkos_view("nonlinear distances",distances));
                } catch (...) {
                    self.fence();
                    throw;
                }
            });
}

void bind_mace_nonlinear_kokkos(py::module_& module)
{
    bind_mace_nonlinear_kokkos_evaluator<MaceNonlinearKokkos>(
        module,"MACENonlinearKokkos");
    bind_mace_nonlinear_kokkos_evaluator<MaceNonlinearFloatKokkos>(
        module,"MACENonlinearKokkosFloat");
}
