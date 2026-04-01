import torch

import triton
import triton.language as tl


class TritonAttention(torch.autograd.Function):

    @staticmethod
    # The context saves the activations so that we can use them to compute the backward pass. (It's essentially a storage area that saves some values that are reused in the backward pass computation.)
    def forward(ctx, Q, K, V, causal, softmax_scale):
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O = torch.empty_like(Q)
        stage = 3 if causal else 1

        # Launch triton programs along two dimensions -> specifies how many programs can be launched in parallel (actual parallelism level is determined by the hardware specification and available resources)
        grid = lambda args: (  # noqa: E731
            # How many blocks of Q we have
            triton.cdiv(
                SEQ_LEN, args["BLOCK_SIZE_Q"]
            ),  # which group of queries are we going to work with
            BATCH_SIZE
            * NUM_HEADS,  # Which head of which batch element are going to work with
            1,
        )

        # M is the logsumexp for the backward pass, one for each query
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

        _att_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            stride_0_batch=Q.stride(0),
            stride_0_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape(2),
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )


def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):  # $
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal(mean=0.0, std=0.5)
        .requires_grad_()
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal(mean=0.0, std=0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal(mean=0.0, std=0.5)
        .requires_grad_()
    )

    softmax_scale = 1.0 / HEAD_DIM**0.5

    d0 = torch.randn_like(Q)

    # reference implementation
    MASK = torch.tril(torch.ones(SEQ_LEN, SEQ_LEN), device="cuda")
    P = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale
    if causal:
        P[:, :, MASK == 0] = float("-inf")

    P = torch.softmax(P, dim=-1).half()

    ref_0 = torch.matmul(P, V)
    ref_0.backward(d0)

    ref_dV, V.grad = V.grad.clone(), None
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None

    # triton implementation
    tri_out = TritonAttention(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(d0)

    tri_dV, V.grad = V.grad.clone(), None
    tri_dK, K.grad = K.grad.clone(), None
    tri_dQ, Q.grad = Q.grad.clone(), None

    # compare
    rtol = 0.0
    atol = 1e-2

    assert torch.allclose(tri_out, ref_0, rtol=rtol, atol=atol)
    assert torch.allclose(tri_dV, ref_dV, rtol=rtol, atol=atol)
    assert torch.allclose(tri_dK, ref_dK, rtol=rtol, atol=atol)
    assert torch.allclose(tri_dQ, ref_dQ, rtol=rtol, atol=atol)
