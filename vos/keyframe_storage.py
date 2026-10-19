"""Optional, exact CPU storage for completed submaps' image tensors.

Install the retrieval wrapper once after constructing a solver. Call the
offload function only after prediction caching, add_points, and graph.optimize
have all completed. The next run_predictions creates its own working submap
on the inference device; retrieved historical frames return to that device
before upstream torch.stack. No model, confidence, or loop decision is changed.

Byte counts describe logical tensor data copied, not CUDA allocator savings:
other references or shared storage may retain an allocation temporarily.
"""
from __future__ import annotations

from functools import wraps


def install_frame_offload(solver):
    """Wrap only this map instance's loop-frame retrieval. Return False if installed.

    Transfers preserve the retrieved tensors' dtype and shape. The target is
    read from the current working submap on every call, rather than captured
    when the hook is installed. The upstream class method remains untouched.
    """
    graph_map = solver.map
    original = graph_map.get_frames_from_loops
    owner = getattr(original, "_vos_frame_offload_solver", None)
    if owner is solver:
        return False
    if owner is not None:
        raise ValueError("This map already has a frame-offload hook for a different solver")

    @wraps(original)
    def get_frames_from_loops(*args, **kwargs):
        retrieved = original(*args, **kwargs)
        if not retrieved:
            return retrieved
        current = solver.current_working_submap.get_all_frames()
        if current is None:
            raise RuntimeError("Loop-frame retrieval needs the current working submap's frames")
        target_device = current.device
        # The pinned GraphMap returns an ordered list, one unbatched image per
        # loop. Blocking copies finish before the upstream stack/model call.
        return [frame.to(device=target_device, non_blocking=False) for frame in retrieved]

    get_frames_from_loops._vos_frame_offload_solver = solver
    graph_map.get_frames_from_loops = get_frames_from_loops
    return True


def offload_stored_frames(solver):
    """Move only completed map submaps' frames to CPU, preserving dtype and values.

    A working submap that is not yet in the map is deliberately excluded. A
    completed working submap already added by add_points is included, as are
    auxiliary loop submaps. CPU tensors retain their original object/storage,
    so repeated calls perform no copies. Transfers complete before returning;
    no CUDA management API or model/retrieval descriptor is accessed, and the
    CUDA allocator cache is not emptied.
    """
    import torch

    diagnostics = dict(submaps_seen=0, submaps_with_frames=0, submaps_copied=0,
                       frames_copied=0, copied_bytes=0, already_cpu_submaps=0,
                       cpu_frame_bytes=0)
    for submap in solver.map.get_submaps():
        diagnostics["submaps_seen"] += 1
        frames = submap.get_all_frames()
        if frames is None:
            continue
        if not isinstance(frames, torch.Tensor):
            raise TypeError("Stored submap frames must be a torch.Tensor or None")
        if frames.ndim < 1:
            raise ValueError("Stored submap frames need a leading frame dimension")
        diagnostics["submaps_with_frames"] += 1
        byte_count = frames.numel() * frames.element_size()
        if frames.device.type == "cpu":
            diagnostics["already_cpu_submaps"] += 1
        else:
            # No dtype, resampling, layout, quantization, or asynchronous-copy
            # option is requested. This is solely a storage-device transfer.
            submap.add_all_frames(frames.to(device="cpu", non_blocking=False))
            diagnostics["submaps_copied"] += 1
            diagnostics["frames_copied"] += int(frames.shape[0])
            diagnostics["copied_bytes"] += int(byte_count)
        diagnostics["cpu_frame_bytes"] += int(byte_count)
    return diagnostics
