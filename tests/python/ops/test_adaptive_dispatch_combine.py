# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License

from types import SimpleNamespace

import pytest
import torch
from mori.ops import AdaptiveEpDispatchCombineOp


class FakeEpOp:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def dispatch(self, input, weights, scales, indices, **kwargs):
        self.calls.append(("dispatch", input.size(0), kwargs))
        return (self.name, "dispatch")

    def combine(self, input, weights, indices, **kwargs):
        self.calls.append(("combine", input.size(0), kwargs))
        return (self.name, "combine")

    def reset(self):
        self.calls.append(("reset",))

    def max_num_tokens_to_recv(self):
        return 256

    def max_num_tokens_to_recv_per_rank(self):
        return 128

    def max_num_tokens_to_send(self):
        return 256

    def max_num_tokens_to_send_per_rank(self):
        return 128


def make_op(threshold=32):
    mori = FakeEpOp("mori")
    kiwi = FakeEpOp("kiwi")
    op = AdaptiveEpDispatchCombineOp(
        SimpleNamespace(),
        kiwi_max_num_tokens=threshold,
        mori_op=mori,
        kiwi_op=kiwi,
    )
    return op, mori, kiwi


def tensors(num_tokens):
    hidden = torch.empty((num_tokens, 8))
    weights = torch.empty((num_tokens, 2))
    indices = torch.empty((num_tokens, 2), dtype=torch.int32)
    return hidden, weights, indices


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    ((0, "kiwi"), (32, "kiwi"), (33, "mori")),
)
def test_complete_pair_uses_bucket_backend(num_tokens, expected):
    op, mori, kiwi = make_op()
    hidden, weights, indices = tensors(num_tokens)

    assert op.dispatch(hidden, weights, None, indices) == (expected, "dispatch")
    assert op.active_backend == expected
    assert op.combine(hidden, None, indices) == (expected, "combine")
    assert op.active_backend is None
    assert op.last_backend == expected

    selected = kiwi if expected == "kiwi" else mori
    unselected = mori if expected == "kiwi" else kiwi
    assert [call[0] for call in selected.calls] == ["dispatch", "combine"]
    assert unselected.calls == []


def test_combine_cannot_switch_when_its_shape_crosses_threshold():
    op, _, kiwi = make_op()
    dispatch_hidden, weights, indices = tensors(16)
    combine_hidden, _, _ = tensors(128)

    op.dispatch(dispatch_hidden, weights, None, indices)
    assert op.combine(combine_hidden, None, indices) == ("kiwi", "combine")
    assert [call[0] for call in kiwi.calls] == ["dispatch", "combine"]


def test_pairing_errors_are_explicit():
    op, _, _ = make_op()
    hidden, weights, indices = tensors(16)

    with pytest.raises(RuntimeError, match="preceding adaptive dispatch"):
        op.combine(hidden, None, indices)

    op.dispatch(hidden, weights, None, indices)
    with pytest.raises(RuntimeError, match="previous combine"):
        op.dispatch(hidden, weights, None, indices)


def test_kiwi_rejects_mori_only_dispatch_features():
    op, _, _ = make_op()
    hidden, weights, indices = tensors(16)

    with pytest.raises(NotImplementedError, match="local expert counts"):
        op.dispatch(
            hidden,
            weights,
            None,
            indices,
            call_local_expert_count=True,
        )
    assert op.active_backend is None


def test_capacity_mismatch_is_rejected():
    mori = FakeEpOp("mori")
    kiwi = FakeEpOp("kiwi")
    kiwi.max_num_tokens_to_recv = lambda: 64

    with pytest.raises(ValueError, match="capacity mismatch"):
        AdaptiveEpDispatchCombineOp(
            SimpleNamespace(),
            kiwi_max_num_tokens=32,
            mori_op=mori,
            kiwi_op=kiwi,
        )
