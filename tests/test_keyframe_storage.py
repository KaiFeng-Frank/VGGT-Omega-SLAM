"""CPU-only storage invariants; CUDA equivalence remains an online gate."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from vos.keyframe_storage import install_frame_offload, offload_stored_frames


class Submap:
    def __init__(self, frames):
        self.frames = frames
        self.retrieval_vectors = object()

    def get_all_frames(self):
        return self.frames

    def add_all_frames(self, frames):
        self.frames = frames

    def get_frame_at_index(self, index):
        return self.frames[index, ...]


class GraphMap:
    def __init__(self, submaps):
        self.submaps = submaps
        self.retrieval_calls = 0

    def get_submaps(self):
        return self.submaps.values()

    def get_frames_from_loops(self, loops):
        self.retrieval_calls += 1
        return [self.submaps[loop.detected_submap_id].get_frame_at_index(loop.detected_submap_frame)
                for loop in loops]


def _solver(submaps, current=None):
    return SimpleNamespace(map=GraphMap(submaps), current_working_submap=current)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_cpu_frames_keep_dtype_values_noncontiguous_layout_and_identity(dtype):
    frames = torch.arange(120).reshape(2, 3, 4, 5).to(dtype).transpose(-1, -2)
    assert not frames.is_contiguous()
    submap = Submap(frames)
    descriptor = submap.retrieval_vectors
    solver = _solver({0: submap}, current=submap)
    first = offload_stored_frames(solver)
    second = offload_stored_frames(solver)
    assert first == second
    assert first == dict(submaps_seen=1, submaps_with_frames=1, submaps_copied=0,
                         frames_copied=0, copied_bytes=0, already_cpu_submaps=1,
                         cpu_frame_bytes=frames.numel() * frames.element_size())
    assert submap.frames is frames
    assert submap.frames.dtype == dtype and submap.frames.stride() == frames.stride()
    assert torch.equal(submap.frames, frames)
    assert submap.retrieval_vectors is descriptor


class _ReportedCudaTensor(torch.Tensor):
    """CPU-backed transfer spy: no CUDA allocation or device initialization.

    Real device-to-host copies are covered by the required online gate. This
    spy exercises selection/counting and observes the transfer arguments.
    """
    @staticmethod
    def __new__(cls, data):
        instance = torch.Tensor._make_subclass(cls, data, require_grad=False)
        instance.transfer_calls = []
        return instance

    @property
    def device(self):
        return torch.device("cuda:0")

    def to(self, **kwargs):
        self.transfer_calls.append(kwargs)
        assert kwargs == {"device": "cpu", "non_blocking": False}
        return self.as_subclass(torch.Tensor).clone(memory_format=torch.preserve_format)


def test_transfer_branch_is_bit_exact_counts_logical_bytes_and_is_idempotent():
    bits = torch.tensor([0, 32768, 16256, 32640, 65408, 32705], dtype=torch.uint16)
    expected = bits.view(torch.bfloat16).reshape(2, 3).T
    source = _ReportedCudaTensor(expected)
    submap = Submap(source)
    # A loop submap and current regular submap can share one map; only frames
    # change, while the working submap not yet in the map stays untouched.
    auxiliary = Submap(_ReportedCudaTensor(torch.arange(12.).reshape(2, 2, 3)))
    working = Submap(_ReportedCudaTensor(torch.ones(4, 3, 2, 2)))
    solver = _solver({0: submap, 17: auxiliary, 19: Submap(None)}, current=working)
    descriptors = [s.retrieval_vectors for s in (submap, auxiliary, working)]
    result = offload_stored_frames(solver)
    assert torch.equal(submap.frames.view(torch.uint16), expected.view(torch.uint16))
    assert submap.frames.dtype == torch.bfloat16 and submap.frames.shape == expected.shape
    assert submap.frames.stride() == expected.stride()
    assert len(source.transfer_calls) == 1
    assert not working.frames.transfer_calls
    assert result["submaps_seen"] == 3 and result["submaps_with_frames"] == 2
    assert result["submaps_copied"] == 2 and result["frames_copied"] == 5
    assert result["copied_bytes"] == result["cpu_frame_bytes"] == 12 + 48
    again = offload_stored_frames(solver)
    assert again["submaps_copied"] == again["copied_bytes"] == 0
    assert again["already_cpu_submaps"] == 2
    assert descriptors == [s.retrieval_vectors for s in (submap, auxiliary, working)]


def test_installed_retrieval_preserves_cpu_values_shapes_order_and_class_method():
    frames = torch.arange(120., dtype=torch.float64).reshape(2, 3, 4, 5).transpose(-1, -2)
    history = Submap(frames)
    solver = _solver({0: history}, current=Submap(torch.ones(2, 3, 5, 4)))
    original = GraphMap.get_frames_from_loops
    assert install_frame_offload(solver)
    installed = solver.map.get_frames_from_loops
    assert not install_frame_offload(solver)
    assert solver.map.get_frames_from_loops is installed
    loops = [SimpleNamespace(detected_submap_id=0, detected_submap_frame=index) for index in (1, 0, 1)]
    retrieved = solver.map.get_frames_from_loops(loops=loops)
    assert solver.map.retrieval_calls == 1 and len(retrieved) == 3
    for image, index in zip(retrieved, (1, 0, 1)):
        assert image.dtype == frames.dtype and image.shape == frames.shape[1:]
        assert image.stride() == frames[index].stride()
        assert image.data_ptr() == frames[index].data_ptr() and torch.equal(image, frames[index])
    assert history.frames is frames and GraphMap.get_frames_from_loops is original
    other = GraphMap({0: history})
    assert other.get_frames_from_loops.__func__ is original


def test_retrieval_uses_current_working_device_dynamically_without_casting_dtype():
    history = Submap(torch.ones(2, 3, 4, 5, dtype=torch.float64))
    solver = _solver({0: history}, current=Submap(torch.empty(2, 3, 4, 5, device="meta", dtype=torch.float16)))
    install_frame_offload(solver)
    loops = [SimpleNamespace(detected_submap_id=0, detected_submap_frame=1)]
    # The meta device tests real torch.to device routing without touching CUDA.
    retrieved = solver.map.get_frames_from_loops(loops)
    assert retrieved[0].device.type == "meta"
    assert retrieved[0].dtype == torch.float64 and retrieved[0].shape == (3, 4, 5)
    assert history.frames.device.type == "cpu"
    solver.current_working_submap = Submap(torch.zeros(1, 3, 4, 5, dtype=torch.float32))
    assert solver.map.get_frames_from_loops(loops)[0].device.type == "cpu"


def test_empty_loops_need_no_current_submap_and_hooks_do_not_change_map_membership():
    solver = _solver({})
    install_frame_offload(solver)
    assert solver.map.get_frames_from_loops([]) == []
    assert solver.map.submaps == {}
    assert offload_stored_frames(solver)["submaps_seen"] == 0
    second_solver = SimpleNamespace(map=solver.map)
    with pytest.raises(ValueError, match="different solver"):
        install_frame_offload(second_solver)


def test_invalid_stored_frames_fail_explicitly():
    with pytest.raises(TypeError, match="torch.Tensor"):
        offload_stored_frames(_solver({0: Submap([1., 2.])}))
    with pytest.raises(ValueError, match="leading frame dimension"):
        offload_stored_frames(_solver({0: Submap(torch.tensor(1.))}))
