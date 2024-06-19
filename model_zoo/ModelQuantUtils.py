import torch
from typing import List


Q_BITS = 4
STORAGE_BITS = 32
PACK_NUM = STORAGE_BITS // Q_BITS


def pseudo_quantize_tensor(w, n_bit=Q_BITS, zero_point=True, group_size=-1):
    org_w_shape = w.shape #(out_features, in_features)
    if group_size > 0:
        assert org_w_shape[-1] % group_size == 0
        w = w.reshape(-1, group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2 ** n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        w = (torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros) * scales
    else:
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = - 2 ** (n_bit - 1)
        scales = max_val / max_int
        zeros = None
        w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

    w = w.reshape(org_w_shape)
    scales = scales.view(w.shape[0], -1)
    zeros = zeros.view(w.shape[0], -1) if zeros is not None else zeros

    return w, scales, zeros


def pack(imatrix: torch.Tensor, direction: str = "column"):
    """
    Packs a 4-bit integer matrix into a packed 32-bit integer matrix.
    Args:
        imatrix (torch.Tensor): matrix of integers
        direction (str): direction of packing, either "column" or "row"

    Returns:
        qmatrix (torch.Tensor): packed matrix of integers
    """
    shifts = torch.arange(0, STORAGE_BITS, Q_BITS, device=imatrix.device)

    imatrix = imatrix.to(torch.int8) & 0x0F

    if direction == "column":
        imatrix = imatrix.view(-1, imatrix.shape[1] // PACK_NUM, PACK_NUM)
        qmatrix = torch.bitwise_left_shift(imatrix, shifts[None, None, :]).sum(dim=-1)

    elif direction == "row":
        imatrix = imatrix.view(imatrix.shape[0] // PACK_NUM, PACK_NUM, -1)
        qmatrix = torch.bitwise_left_shift(imatrix, shifts[None, :, None]).sum(dim=1)

    qmatrix = qmatrix.to(torch.int32)

    return qmatrix


def unpack(qmatrix: torch.Tensor, direction: str = "column"):
    """
    Unpacks a 32-bit packed integer matrix into a 4-bit integer matrix.

    Args:
        qmatrix (torch.Tensor): matrix of packed integers
        direction (str): direction of unpacking, either "column" or "row"

    Returns:
        imatrix (torch.Tensor): matrix of integers
    """
    shifts = torch.arange(0, STORAGE_BITS, Q_BITS, device=qmatrix.device)

    if direction == "column":
        imatrix = torch.bitwise_right_shift(
            qmatrix[:, :, None], shifts[None, None, :]
        ).view(qmatrix.shape[0], -1)

    elif direction == "row":
        imatrix = torch.bitwise_right_shift(
            qmatrix[:, None, :], shifts[None, :, None]
        ).view(-1, qmatrix.shape[-1])

    imatrix = imatrix.to(torch.int8) & 0x0F  # eventually correct overflow

    return imatrix


def quantize_int4(fmatrix, scales, zeros, group_size, n_bits=Q_BITS):
    """
    Quantizes a matrix of 16-bit floats into a matrix of unsigned 4 bit integers.

    Args:
        fmatrix (torch.Tensor): matrix of 16-bit floats
        scales (torch.Tensor): matrix of 16-bit floats
        zeros (torch.Tensor): matrix of 4 bit integers
        group_size (int): group size

    Returns:
        imatrix (torch.Tensor): matrix of unsigned 4 bit integers
    """

    if zeros is None:
        offset = 2 ** (n_bits - 1)
    else:
        zeros = zeros.to(torch.int8) & 0x0F
        offset = zeros.repeat_interleave(group_size, dim=1)

    imatrix = torch.round(
        fmatrix / scales.repeat_interleave(group_size, dim=1) + offset)

    imatrix = imatrix.to(torch.uint8) & 0x0F

    return imatrix


def dequantize_int4(imatrix, scales, zeros, group_size, n_bits=Q_BITS):
    """
    Dequantizes a unsigned 4/8 bit integer matrix into a float matrix.

    Args:
        imatrix (torch.Tensor): matrix of unsigned 4 bit integers
        scales (torch.Tensor): matrix of 16-bit floats
        zeros (torch.Tensor): matrix of 4-bit integers
        group_size (int): group size

    Returns:
        fmatrix (torch.Tensor): matrix of 16-bit floats
    """
    if zeros is None:
        offset = 2 ** (n_bits - 1)
    else:
        zeros = zeros.to(torch.int8) & 0x0F
        offset = zeros.repeat_interleave(group_size, dim=1)
    imatrix = imatrix.to(torch.int8) & 0x0F

    fmatrix = (imatrix - offset) * scales.repeat_interleave(group_size, dim=1)
    fmatrix = fmatrix.to(torch.float16)

    return fmatrix


if __name__ == "__main__":
    layer = torch.nn.Linear(in_features=512, out_features=2048, dtype=torch.float16)
    weight = layer.weight #shape -> (output_features, in_features)
    qdq_w, scale, zp = pseudo_quantize_tensor(weight, n_bit=Q_BITS, zero_point=False, group_size=128)
    int4_w = quantize_int4(qdq_w, scale, zp, 128)
    packed_int32_w = pack(int4_w)
    unpacked_int4_w = unpack(packed_int32_w)
    qdq_w_after_pack = dequantize_int4(unpacked_int4_w, scale, zp, 128)
