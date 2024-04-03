import torch
from torch.nn import functional as F

# get the channel slice for certrain output
# class Converter_key2channel(object):
#      def __init__(self, keys, channels):
#          super(Converter_key2channel, self).__init__()
#          self.keys = keys
#          self.channels = channels

#      def __call__(self, key):
#         # find the corresponding index
#         index = self.keys.index(key)

#         s = sum(self.channels[:index])
#         e = s + self.channels[index]

#         return slice(s, e, 1)

# get the channel slice for certrain output

class Converter_key2channel(object):
     def __init__(self, keys, channels):
         super(Converter_key2channel, self).__init__()
         
         # flatten keys and channels
         self.keys = [key for key_group in keys for key in key_group]
         self.channels = [channel for channel_groups in channels for channel in channel_groups]

     def __call__(self, key):
        # find the corresponding index
        index = self.keys.index(key)

        s = sum(self.channels[:index])
        e = s + self.channels[index]

        return slice(s, e, 1)

def sigmoid_hm(hm_features):
    x = hm_features.sigmoid_()
    x = x.clamp(min=1e-4, max=1 - 1e-4)

    return x

def nms_hm(heat_map, kernel=3, reso=1):
    kernel = int(kernel / reso)
    if kernel % 2 == 0:
        kernel += 1
    
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat_map,
                        kernel_size=(kernel, kernel),
                        stride=1,
                        padding=pad)

    eq_index = (hmax == heat_map).float()

    return heat_map * eq_index


def select_topk(heat_map, K=100):
    '''
    Args:
        heat_map: heat_map in [N, C, H, W]
        K: top k samples to be selected
        score: detection threshold

    Returns:

    '''
    batch, cls, height, width = heat_map.size()

    # First select topk scores in all classes and batchs
    # [N, C, H, W] -----> [N, C, H*W]
    heat_map = heat_map.view(batch, cls, -1)
    # Both in [N, C, K]
    topk_scores_all, topk_inds_all = torch.topk(heat_map, K)

    # topk_inds_all = topk_inds_all % (height * width) # todo: this seems redudant
    topk_ys = (topk_inds_all / width).float()
    topk_xs = (topk_inds_all % width).float()

    assert isinstance(topk_xs, torch.cuda.FloatTensor)
    assert isinstance(topk_ys, torch.cuda.FloatTensor)

    # Select topK examples across channel (classes)
    # [N, C, K] -----> [N, C*K]
    topk_scores_all = topk_scores_all.view(batch, -1)
    # Both in [N, K]
    topk_scores, topk_inds = torch.topk(topk_scores_all, K)
    topk_clses = (topk_inds / K).float()

    assert isinstance(topk_clses, torch.cuda.FloatTensor)

    # First expand it as 3 dimension
    topk_inds_all = _gather_feat(topk_inds_all.view(batch, -1, 1), topk_inds).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_inds).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_inds).view(batch, K)

    return topk_scores, topk_inds_all, topk_clses, topk_ys, topk_xs


def _gather_feat(feat, ind):
    '''
    Select specific indexs on feature map
    Args:
        feat: all results in 3 dimensions
        ind: positive index

    Returns:

    '''
    channel = feat.size(-1)                                            
    ind = ind.unsqueeze(-1).expand(ind.size(0), ind.size(1), channel)
    feat = feat.gather(1, ind)

    return feat


def select_point_of_interest(batch, index, feature_maps):
    '''
    Select POI(point of interest) on feature map
    Args:
        batch: batch size
        index: in point format or index format
        feature_maps: regression feature map in [N, C, H, W]

    Returns:

    '''
    w = feature_maps.shape[3]
    if len(index.shape) == 3:
        index = index[:, :, 1] * w + index[:, :, 0]
    index = index.view(batch, -1)
    # [N, C, H, W] -----> [N, H, W, C]
    feature_maps = feature_maps.permute(0, 2, 3, 1).contiguous()
    channel = feature_maps.shape[-1]
    # [N, H, W, C] -----> [N, H*W, C]
    feature_maps = feature_maps.view(batch, -1, channel)
    # expand index in channels
    index = index.unsqueeze(-1).repeat(1, 1, channel)
    # select specific features bases on POIs
    feature_maps = feature_maps.gather(1, index.long())

    return feature_maps

def generate_grid(h, w):
    x = torch.arange(0, h)
    y = torch.arange(0, w)
    grid = torch.stack([x.repeat(w), y.repeat(h,1).t().contiguous().view(-1)],1)
    return grid

def soft_get(reg, pts, force_get=False, force_get_uncertainty=False):
    pts_x = pts[:, 0]
    pts_y = pts[:, 1]

    h, w = reg.shape[-2], reg.shape[-1]
    # if force_get:
    #     pts_x = pts_x.clamp(0, w-1)
    #     pts_y = pts_y.clamp(0, h-1)

    pts_x_low = pts_x.floor().long()
    pts_x_high = pts_x.ceil().long()

    pts_y_low = pts_y.floor().long()
    pts_y_high = pts_y.ceil().long()

    valid_idx = (pts_y_low >= 0) & (pts_y_high < h) & (pts_x_low >= 0) & (pts_x_high < w)

    pts_x_low_valid = pts_x_low[valid_idx]
    pts_x_high_valid = pts_x_high[valid_idx]
    pts_y_low_valid = pts_y_low[valid_idx]
    pts_y_high_valid = pts_y_high[valid_idx]

    pts_x = pts_x[valid_idx]
    pts_y = pts_y[valid_idx]

    rop_lt = reg[..., pts_y_low_valid, pts_x_low_valid]
    rop_rt = reg[..., pts_y_low_valid, pts_x_high_valid]
    rop_ld = reg[..., pts_y_high_valid, pts_x_low_valid]
    rop_rd = reg[..., pts_y_high_valid, pts_x_high_valid]

    rop_t = (1 - pts_x + pts_x_low_valid) * rop_lt + (1 - pts_x_high_valid + pts_x) * rop_rt
    rop_d = (1 - pts_x + pts_x_low_valid) * rop_ld + (1 - pts_x_high_valid + pts_x) * rop_rd

    rop = (1 - pts_y + pts_y_low_valid) * rop_t + (1 - pts_y_high_valid + pts_y) * rop_d
    if force_get and not torch.all(valid_idx):
        shape = list(rop.shape)
        shape[-1] = len(valid_idx)
        rop_force = torch.zeros(*shape, dtype=rop.dtype, device=rop.device)
        rop_force[..., valid_idx] = rop

        pts_invalid = pts[~valid_idx]

        grid = generate_grid(w, h).to(pts_invalid.device)
        reg_invalid_all = []
        for pts_in in pts_invalid:
            diff = (pts_in - grid)
            dis = diff[:, 0] ** 2 + diff[:, 1] ** 2
            closest_idx = dis.argmin()

            reg_invalid_all.append(reg[..., grid[closest_idx, 1].long(), grid[closest_idx, 0].long()])
        reg_invalid_all = torch.stack(reg_invalid_all, dim=-1)
        rop_force[..., ~valid_idx] = reg_invalid_all.detach()
        if force_get_uncertainty:
            # import pdb; pdb.set_trace()
            # assert rop_force.shape[0] == 2
            if rop_force.shape[0] == 2:
                rop_force[1, ~valid_idx] = 30
            elif rop_force.shape[0] == 6:
                rop_force[1, ~valid_idx] = 30
                rop_force[3, ~valid_idx] = 30
                rop_force[5, ~valid_idx] = 30
            else:
                raise NotImplementedError
        return rop_force, valid_idx

    return rop, valid_idx