"""Native CUDA variable-length attention without runtime length synchronization."""
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _has_native_varlen():
    """An exported ATen symbol alone does not imply a compiled Flash backend."""
    try:
        forward = torch.ops.aten._flash_attention_forward.default
        args = tuple(arg.name for arg in forward._schema.arguments)
        return (args[:10] == ('query', 'key', 'value', 'cum_seq_q', 'cum_seq_k',
                             'max_q', 'max_k', 'dropout_p', 'is_causal', 'return_debug_mask')
                and torch.backends.cuda.is_flash_attention_available()
                and torch.version.hip is None
                and all(torch._C._dispatch_has_kernel_for_dispatch_key(op, dispatch)
                        for op, dispatch in (
                            ('aten::_flash_attention_forward', 'CUDA'),
                            ('aten::_flash_attention_backward', 'CUDA')))
                # Generated VariableType uses the Autograd alias on standard
                # builds; querying only the concrete key incorrectly says no.
                and any(torch._C._dispatch_has_kernel_for_dispatch_key(
                    'aten::_flash_attention_forward', key)
                    for key in ('AutogradCUDA', 'Autograd')))
    except (AttributeError, RuntimeError):
        return False


@lru_cache(maxsize=None)
def _device_capability(device):
    return torch.cuda.get_device_capability(device)


def _compute_dtype(query):
    if torch.is_autocast_enabled('cuda'):
        return torch.get_autocast_dtype('cuda')
    return query.dtype


def can_use_varlen_flash(query, key, value, *, dropout_p=0.0):
    """Check metadata only; no GPU reductions, host copies, or trial kernels.

    This adapter handles equal Q/K/V head widths, as used by the Decoder.
    Unsupported backends, dtypes, dimensions and training shapes use SDPA.
    """
    if (query.device.type != 'cuda' or not _has_native_varlen()
            or not torch.backends.cuda.flash_sdp_enabled()):
        return False
    if any(t.device != query.device or t.ndim != 3 for t in (query, key, value)):
        return False
    if (query.shape[1:] != key.shape[1:] or key.shape != value.shape
            or query.shape[0] == 0 or key.shape[0] == 0 or query.shape[1] == 0):
        return False
    depth = query.shape[-1]
    if not (0 < depth <= 256 and depth % 8 == 0 and 0 <= dropout_p < 1):
        return False
    dtype = _compute_dtype(query)
    if dtype not in (torch.float16, torch.bfloat16):
        return False
    if torch.is_autocast_enabled('cuda'):
        if any(t.dtype not in (torch.float16, torch.bfloat16, torch.float32)
               for t in (query, key, value)):
            return False
    elif any(t.dtype != dtype for t in (key, value)):
        return False
    capability = _device_capability(query.device)
    if not (8, 0) <= capability <= (12, 1):
        return False
    # Match native backward's restrictions on consumer Ampere/Ada/Blackwell.
    limited = (8, 6) <= capability <= (8, 9) or (12, 0) <= capability <= (12, 1)
    if (limited and torch.is_grad_enabled()
            and any(t.requires_grad for t in (query, key, value))
            and (192 < depth <= 224 or depth > 224 and dropout_p > 0)):
        return False
    return True


def varlen_flash_attention(query, key, value, cu_q, cu_k, max_q, max_k, *,
                           dropout_p, is_causal, scale):
    """Call the ATen operator with its registered backward and explicit scale.

    Cumulative lengths are prepared on CPU alongside the packed row layout,
    then transferred with it. Their contents are deliberately not read here.
    Causal callers must supply matching per-sequence Q/K lengths (self-attention).
    """
    if (type(max_q) is not int or type(max_k) is not int or min(max_q, max_k) <= 0
            or cu_q.ndim != 1 or cu_q.shape != cu_k.shape or cu_q.numel() < 2
            or any(c.dtype != torch.int32 or c.device != query.device
                   for c in (cu_q, cu_k))):
        raise ValueError('varlen attention requires device int32 cumulative lengths and Python maxima')
    original_dtype = query.dtype
    dtype = _compute_dtype(query)
    # The private operator has no autocast conversion of its own. Contiguous
    # packing can copy strided projections; it is not an allocation-free API.
    with torch.autocast(device_type=query.device.type, enabled=False):
        q, k, v = (t.to(dtype).contiguous() for t in (query, key, value))
        output = torch.ops.aten._flash_attention_forward(
            q, k, v, cu_q.contiguous(), cu_k.contiguous(), max_q, max_k,
            dropout_p, is_causal, False, scale=scale)[0]
    return output.to(original_dtype)
