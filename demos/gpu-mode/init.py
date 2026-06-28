# init.py (triton seed)  |  parent: none
# status: PASS  |  geomean: 61.1692 ms
# trick: naive unblocked Householder QR in Triton, one program per matrix;
#        custom path for n in {32,176,352,512}, torch.geqrf fallback otherwise.
import sys, io
if sys.stdout is None: sys.stdout = io.StringIO()
if sys.stderr is None: sys.stderr = io.StringIO()

from task import input_t, output_t
import torch
import triton
import triton.language as tl


@triton.jit
def qr_naive_kernel(H_ptr, Tau_ptr, n,
                    stride_b, stride_i, stride_j, stride_tb,
                    BLOCK_M: tl.constexpr, BLOCK_J: tl.constexpr):
    # One Triton program factors one matrix of the batch, in place in H,
    # by an unblocked (column-by-column) Householder QR producing the exact
    # torch.geqrf packing: R in the upper triangle, reflector v_k below the
    # diagonal of column k (with implicit 1 on the diagonal), tau in Tau.
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    row_active = offs_m < n
    mat = pid * stride_b

    # `n` is a runtime arg (not constexpr) so this is a real loop, not unrolled.
    for k in range(0, n):
        # current (trailing-updated) column k, rows 0..n-1
        col_ptr = H_ptr + mat + offs_m * stride_i + k * stride_j
        v = tl.load(col_ptr, mask=row_active, other=0.0)

        is_diag = offs_m == k
        below = (offs_m > k) & row_active

        # alpha = H[k,k]; sigma = sum of squares strictly below the diagonal
        alpha = tl.sum(tl.where(is_diag, v, 0.0), axis=0)
        sigma = tl.sum(tl.where(below, v * v, 0.0), axis=0)

        # dlarfg: beta = -sign(alpha)*||x||, tau = (beta-alpha)/beta,
        # v_below = x_below/(alpha-beta), v_diag = 1. No reflection when sigma==0.
        has_refl = sigma > 0.0
        xnorm = tl.sqrt(alpha * alpha + sigma)
        sign = tl.where(alpha >= 0.0, 1.0, -1.0)
        beta_c = -sign * xnorm
        tau_k = tl.where(has_refl, (beta_c - alpha) / beta_c, 0.0)
        inv = tl.where(has_refl, 1.0 / (alpha - beta_c), 0.0)
        beta = tl.where(has_refl, beta_c, alpha)

        # Householder vector with the implicit 1 on the diagonal
        vvec = tl.where(is_diag, 1.0, tl.where(below, v * inv, 0.0))

        # store packed column k (beta on the diagonal, reflector below);
        # rows < k hold finalized R entries and must not be touched.
        store_val = tl.where(is_diag, beta, v * inv)
        store_mask = (offs_m >= k) & row_active
        tl.store(col_ptr, store_val, mask=store_mask)

        # apply (I - tau * vvec * vvec^T) to every trailing column j > k
        for jb in range(0, BLOCK_M, BLOCK_J):
            offs_j = jb + tl.arange(0, BLOCK_J)
            col_active = (offs_j > k) & (offs_j < n)
            t_ptr = (H_ptr + mat
                     + offs_m[:, None] * stride_i
                     + offs_j[None, :] * stride_j)
            t_mask = row_active[:, None] & col_active[None, :]
            T = tl.load(t_ptr, mask=t_mask, other=0.0)
            w = tl.sum(vvec[:, None] * T, axis=0)            # w_j = vvec^T a_j
            T_new = T - tau_k * (vvec[:, None] * w[None, :])  # a_j -= tau w_j vvec
            tl.store(t_ptr, T_new, mask=t_mask)

        tl.store(Tau_ptr + pid * stride_tb + k, tau_k)


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


# Shapes routed to the custom Triton kernel. The dominant (640,512) lever lives
# here; n in {1024,2048,4096} fall back to cuSOLVER for now (an obvious first
# optimization target for later iterations).
_CUSTOM_N = {32, 176, 352, 512}


def custom_kernel(data: input_t) -> output_t:
    A = data                       # (batch, n, n) float32 cuda — read-only
    b, n, _ = A.shape
    if n not in _CUSTOM_N:
        return torch.geqrf(A)      # cuSOLVER is strong / kernel not specialized

    H = A.clone()                  # never mutate the input; factor the copy
    Tau = torch.empty((b, n), dtype=A.dtype, device=A.device)
    # Pad to the next power of two STRICTLY above n, so there is always at least
    # one masked guard row. An exact-fit row tile (BLOCK_M == n, hit at n=32 and
    # n=512) miscompiles the axis-0 reduction and corrupts the trailing update;
    # keeping a padding lane avoids that layout entirely.
    BLOCK_M = _next_pow2(n + 1)
    BLOCK_J = 16 if BLOCK_M >= 512 else 32
    num_warps = 8 if BLOCK_M >= 512 else 4
    qr_naive_kernel[(b,)](
        H, Tau, n,
        H.stride(0), H.stride(1), H.stride(2), Tau.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_J=BLOCK_J, num_warps=num_warps,
    )
    return H, Tau
