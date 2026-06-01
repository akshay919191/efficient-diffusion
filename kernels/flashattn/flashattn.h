#pragma once

#include <torch/extension.h>

std::vector<torch::Tensor> flash_attn_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    auto b = q.size(0);
    auto h = q.size(1);
    auto s = q.size(2);
    auto d = q.size(3);

    auto out = torch::zeros_like(q);
    auto lse = torch::zeros({b, h, s}, q.options().dtype(torch::kFloat32));

    const int Br = 64;
    dim3 grid(b, h, s / Br);
    dim3 block(128);
    size_t smem = 48 * 1024;

    if (d == 64) {
        forwardFlashAttention<64, 64, 32, 8><<<grid, block, smem>>>(
            (const half*)q.data_ptr<at::Half>(),
            (const half*)k.data_ptr<at::Half>(),
            (const half*)v.data_ptr<at::Half>(),
            (half*)out.data_ptr<at::Half>(),
            lse.data_ptr<float>(),
            0, 0,
            static_cast<int>(s),
            static_cast<int>(d)
        );
    } else {
        forwardFlashAttention<32, 64, 32, 8><<<grid, block, smem>>>(
            (const half*)q.data_ptr<at::Half>(),
            (const half*)k.data_ptr<at::Half>(),
            (const half*)v.data_ptr<at::Half>(),
            (half*)out.data_ptr<at::Half>(),
            lse.data_ptr<float>(),
            0, 0,
            static_cast<int>(s),
            static_cast<int>(d)
        );
    }

    return {out, lse};
}

std::vector<torch::Tensor> flash_attn_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor o, torch::Tensor lse, torch::Tensor d_out
) {
    auto b = q.size(0);
    auto h = q.size(1);
    auto s = q.size(2);
    auto d = q.size(3);

    auto dq = torch::zeros_like(q);
    auto dk = torch::zeros_like(k);
    auto dv = torch::zeros_like(v);

    auto score_buf = torch::zeros({b, h, 32, static_cast<long>(d)}, q.options());

    const int Br = 32;
    dim3 grid(b, h, s / Br);
    dim3 block(128);
    size_t smem = 48 * 1024;

    if (d == 64) {
        backwardFlashAttention_DQ<32, 64, 8><<<grid, block, smem>>>(
            (const half*)q.data_ptr<at::Half>(),
            (const half*)k.data_ptr<at::Half>(),
            (const half*)v.data_ptr<at::Half>(),
            (const half*)o.data_ptr<at::Half>(),
            lse.data_ptr<float>(),
            (half*)score_buf.data_ptr<at::Half>(),
            (half*)d_out.data_ptr<at::Half>(),
            (half*)dq.data_ptr<at::Half>(),
            static_cast<int>(s),
            static_cast<int>(d)
        );
        backwardFlashAttention_DK_DV<32, 64, 8><<<grid, block, smem>>>(
            (const half*)q.data_ptr<at::Half>(),
            (const half*)k.data_ptr<at::Half>(),
            (const half*)v.data_ptr<at::Half>(),
            (const half*)o.data_ptr<at::Half>(),
            lse.data_ptr<float>(),
            (half*)score_buf.data_ptr<at::Half>(),
            (half*)d_out.data_ptr<at::Half>(),
            (half*)dk.data_ptr<at::Half>(),
            (half*)dv.data_ptr<at::Half>(),
            static_cast<int>(s),
            static_cast<int>(d)
        );
    } else {
        backwardFlashAttention_DQ<32, 32, 8><<<grid, block, smem>>>(
            (const half*)q.data_ptr<at::Half>(),
            (const half*)k.data_ptr<at::Half>(),
            (const half*)v.data_ptr<at::Half>(),
            (const half*)o.data_ptr<at::Half>(),
            lse.data_ptr<float>(),
            (half*)score_buf.data_ptr<at::Half>(),
            (half*)d_out.data_ptr<at::Half>(),
            (half*)dq.data_ptr<at::Half>(),
            static_cast<int>(s),
            static_cast<int>(d)
        );
        backwardFlashAttention_DK_DV<32, 32, 8><<<grid, block, smem>>>(
            (const half*)q.data_ptr<at::Half>(),
            (const half*)k.data_ptr<at::Half>(),
            (const half*)v.data_ptr<at::Half>(),
            (const half*)o.data_ptr<at::Half>(),
            lse.data_ptr<float>(),
            (half*)score_buf.data_ptr<at::Half>(),
            (half*)d_out.data_ptr<at::Half>(),
            (half*)dk.data_ptr<at::Half>(),
            (half*)dv.data_ptr<at::Half>(),
            static_cast<int>(s),
            static_cast<int>(d)
        );
    }

    return {dq, dk, dv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward",  &flash_attn_forward,  "Flash Attention Forward");
    m.def("backward", &flash_attn_backward, "Flash Attention Backward");
}