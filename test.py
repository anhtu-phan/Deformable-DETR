import torch
import torch.nn.functional as F


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    sampling_ = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (sampling_ * attention_weights)
    output = output.sum(-1).view(N_, M_*D_, Lq_)
    output = output.transpose(1, 2).contiguous()
    return output


if __name__ == '__main__':
    # [batch, len_k, n_head, d]
    v = torch.rand((2, 5, 1, 4))
    # [n_level, 2(H,W)]
    value_spatial = torch.tensor([[1, 5]])
    # [batch, len_q, n_head, n_level, n_point, 2]
    sampling = torch.randn((2, 4, 1, 1, 3, 2))
    # [batch, len_q, n_head, n_level, n_point]
    attention = torch.randn((2, 4, 1, 1, 3))
    print(ms_deform_attn_core_pytorch(v, value_spatial, sampling, attention))
