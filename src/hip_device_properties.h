#ifndef BC_HIP_DEVICE_PROPERTIES_H
#define BC_HIP_DEVICE_PROPERTIES_H
#include <stdint.h>
#include <stddef.h>
/* hipDeviceProp_t ABI R0600, through gcnArchName. ROCm versions the function
 * that writes this struct specifically so a header-free loader can pin its
 * layout. Only name, totalGlobalMem and gcnArchName are read. */
typedef struct {
	char name[256]; unsigned char uuid[16]; char luid[8];
	uint32_t luid_device_node_mask; size_t total_global_mem;
	size_t shared_mem_per_block; int32_t regs_per_block, warp_size;
	size_t mem_pitch; int32_t max_threads_per_block, max_threads_dim[3];
	int32_t max_grid_size[3], clock_rate; size_t total_const_mem;
	int32_t major, minor; size_t texture_alignment, texture_pitch_alignment;
	int32_t device_overlap, compute_mode_dummy[4];
	int32_t max_texture_1d, max_texture_1d_mipmap, max_texture_1d_linear;
	int32_t max_texture_2d[2], max_texture_2d_mipmap[2];
	int32_t max_texture_2d_linear[3], max_texture_2d_gather[2];
	int32_t max_texture_3d[3], max_texture_3d_alt[3];
	int32_t max_texture_cubemap, max_texture_1d_layered[2];
	int32_t max_texture_2d_layered[3], max_texture_cubemap_layered[2];
	int32_t max_surface_1d, max_surface_2d[2], max_surface_3d[3];
	int32_t max_surface_1d_layered[2], max_surface_2d_layered[3];
	int32_t max_surface_cubemap, max_surface_cubemap_layered[2];
	size_t surface_alignment;
	int32_t concurrent_kernels, ecc_enabled, pci_bus_id, pci_device_id;
	int32_t pci_domain_id, tcc_driver, async_engine_count, unified_addressing;
	int32_t memory_clock_rate, memory_bus_width, l2_cache_size;
	int32_t persisting_l2_cache_max_size, max_threads_per_multi_processor;
	int32_t stream_priorities_supported, global_l1_cache_supported;
	int32_t local_l1_cache_supported; size_t shared_mem_per_multiprocessor;
	int32_t regs_per_multiprocessor, managed_memory, is_multi_gpu_board;
	int32_t multi_gpu_board_group_id, host_native_atomic_supported;
	int32_t single_to_double_precision_perf_ratio, pageable_memory_access;
	int32_t concurrent_managed_access, compute_preemption_supported;
	int32_t can_use_host_pointer_for_registered_mem, cooperative_launch;
	int32_t cooperative_multi_device_launch; size_t shared_mem_per_block_optin;
	int32_t pageable_memory_access_uses_host_page_tables;
	int32_t direct_managed_mem_access_from_host, max_blocks_per_multiprocessor;
	int32_t access_policy_max_window_size; size_t reserved_shared_mem_per_block;
	int32_t host_register_supported, sparse_hip_array_supported;
	int32_t host_register_read_only_supported;
	int32_t timeline_semaphore_interop_supported, memory_pools_supported;
	int32_t gpu_direct_rdma_supported;
	uint32_t gpu_direct_rdma_flush_writes_options;
	int32_t gpu_direct_rdma_writes_ordering;
	uint32_t memory_pool_supported_handle_types;
	int32_t deferred_mapping_hip_array_supported, ipc_event_supported;
	int32_t cluster_launch, unified_function_pointers;
	int32_t reserved[63], hip_reserved[32];
	char gcn_arch_name[256];
	unsigned char trailing_pad[512];
} bc_hip_device_prop;

#endif
