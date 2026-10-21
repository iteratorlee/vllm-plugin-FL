# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest


def test_native_gdn_skips_only_qwen_triton_warmup(monkeypatch):
    pytest.importorskip("torch_npu")
    from vllm.model_executor.custom_op import PluggableLayer
    from vllm.model_executor.warmup import kernel_warmup

    from vllm_fl.dispatch.backends.vendor.ascend.patch import patch_gdn

    original_kernel_warmup = kernel_warmup.kernel_warmup
    original_qwen_warmup = Mock(side_effect=AssertionError("Triton is unused"))
    monkeypatch.setattr(kernel_warmup, "qwen_triton_warmup", original_qwen_warmup)
    register = Mock()
    monkeypatch.setattr(PluggableLayer, "register_oot", register)
    patch_gdn()
    kernel_warmup.qwen_triton_warmup(None, None)
    original_qwen_warmup.assert_not_called()
    register.assert_called_once()
    assert kernel_warmup.kernel_warmup is original_kernel_warmup
