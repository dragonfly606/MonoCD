import cv2
import numpy as np
import pdb
import torch
import torch.nn.functional as F

import torchvision.ops.roi_align as roi_align
from data.datasets.kitti_utils import convertAlpha2Rot
PI = np.pi

from sklearn import linear_model

class Anno_Encoder():
		def __init__(self, cfg):
			device = cfg.MODEL.DEVICE
			self.INF = 100000000
			self.EPS = 1e-3

			# center related
			self.num_cls = len(cfg.DATASETS.DETECT_CLASSES)
			self.min_radius = cfg.DATASETS.MIN_RADIUS
			self.max_radius = cfg.DATASETS.MAX_RADIUS
			self.center_ratio = cfg.DATASETS.CENTER_RADIUS_RATIO
			self.target_center_mode = cfg.INPUT.HEATMAP_CENTER
			# if mode == 'max', centerness is the larger value, if mode == 'area', assigned to the smaller bbox
			self.center_mode = cfg.MODEL.HEAD.CENTER_MODE
			
			# depth related
			self.depth_mode = cfg.MODEL.HEAD.DEPTH_MODE
			self.depth_range = cfg.MODEL.HEAD.DEPTH_RANGE
			self.y3d_range = cfg.Y3D_RANGE
			self.depth_ref = torch.as_tensor(cfg.MODEL.HEAD.DEPTH_REFERENCE).to(device=device)

			# dimension related
			self.dim_mean = torch.as_tensor(cfg.MODEL.HEAD.DIMENSION_MEAN).to(device=device)
			self.dim_std = torch.as_tensor(cfg.MODEL.HEAD.DIMENSION_STD).to(device=device)
			self.dim_modes = cfg.MODEL.HEAD.DIMENSION_REG

			# y3d related
			self.y3d_mean = torch.as_tensor(cfg.MODEL.HEAD.Y_MEAN).to(device=device)
			self.y3d_std = torch.as_tensor(cfg.MODEL.HEAD.Y_STD).to(device=device)
			self.y3d_modes = cfg.Y3D_REG
			self.use_y3d_alone = cfg.USE_Y3D_ALONE
			self.compute_depth_only_from_roof = cfg.COMPUTE_DEPTH_ONLY_FROM_ROOF
			self.use_edge_slope = cfg.MODEL.HEAD.USE_EDGE_SLOPE
			self.horizon_fitting_method = cfg.MODEL.HEAD.HORIZON_FITTING_METHOD

			# orientation related
			self.alpha_centers = torch.tensor([0, PI / 2, PI, - PI / 2]).to(device=device)
			self.multibin = (cfg.INPUT.ORIENTATION == 'multi-bin')
			self.orien_bin_size = cfg.INPUT.ORIENTATION_BIN_SIZE

			# offset related
			self.offset_mean = cfg.MODEL.HEAD.REGRESSION_OFFSET_STAT[0]
			self.offset_std = cfg.MODEL.HEAD.REGRESSION_OFFSET_STAT[1]

			# output info
			self.down_ratio = cfg.MODEL.BACKBONE.DOWN_RATIO
			self.output_height = cfg.INPUT.HEIGHT_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
			self.output_width = cfg.INPUT.WIDTH_TRAIN // cfg.MODEL.BACKBONE.DOWN_RATIO
			self.K = self.output_width * self.output_height

		@staticmethod
		def rad_to_matrix(rotys, N):
			device = rotys.device

			cos, sin = rotys.cos(), rotys.sin()

			i_temp = torch.tensor([[1, 0, 1],
								 [0, 1, 0],
								 [-1, 0, 1]]).to(dtype=torch.float32, device=device)

			ry = i_temp.repeat(N, 1).view(N, -1, 3)

			ry[:, 0, 0] *= cos
			ry[:, 0, 2] *= sin
			ry[:, 2, 0] *= sin
			ry[:, 2, 2] *= cos

			return ry

		def decode_box2d_fcos(self, centers, pred_offset, pad_size=None, out_size=None):
			box2d_center = centers.view(-1, 2)
			box2d = box2d_center.new(box2d_center.shape[0], 4).zero_()
			# left, top, right, bottom
			box2d[:, :2] = box2d_center - pred_offset[:, :2]
			box2d[:, 2:] = box2d_center + pred_offset[:, 2:]

			# for inference
			if pad_size is not None:
				N = box2d.shape[0]
				out_size = out_size[0]
				# upscale and subtract the padding
				box2d = box2d * self.down_ratio - pad_size.repeat(1, 2)
				# clamp to the image bound
				box2d[:, 0::2].clamp_(min=0, max=out_size[0].item() - 1)
				box2d[:, 1::2].clamp_(min=0, max=out_size[1].item() - 1)

			return box2d

		def encode_box3d(self, rotys, dims, locs):
			'''
			construct 3d bounding box for each object.
			Args:
					rotys: rotation in shape N
					dims: dimensions of objects
					locs: locations of objects

			Returns:

			'''
			if len(rotys.shape) == 2:
					rotys = rotys.flatten()
			if len(dims.shape) == 3:
					dims = dims.view(-1, 3)
			if len(locs.shape) == 3:
					locs = locs.view(-1, 3)

			device = rotys.device
			N = rotys.shape[0]
			ry = self.rad_to_matrix(rotys, N)

			# l, h, w
			dims_corners = dims.view(-1, 1).repeat(1, 8)
			dims_corners = dims_corners * 0.5
			dims_corners[:, 4:] = -dims_corners[:, 4:]
			index = torch.tensor([[4, 5, 0, 1, 6, 7, 2, 3],
								[0, 1, 2, 3, 4, 5, 6, 7],
								[4, 0, 1, 5, 6, 2, 3, 7]]).repeat(N, 1).to(device=device)
			
			box_3d_object = torch.gather(dims_corners, 1, index)
			box_3d = torch.matmul(ry, box_3d_object.view(N, 3, -1))
			box_3d += locs.unsqueeze(-1).repeat(1, 1, 8)

			return box_3d.permute(0, 2, 1)

		def decode_depth(self, depths_offset):
			'''
			Transform depth offset to depth
			'''
			if self.depth_mode == 'exp':
				depth = depths_offset.exp()
			elif self.depth_mode == 'linear':
				depth = depths_offset * self.depth_ref[1] + self.depth_ref[0]
			elif self.depth_mode == 'inv_sigmoid':
				depth = 1 / torch.sigmoid(depths_offset) - 1
			else:
				raise ValueError

			if self.depth_range is not None:
				depth = torch.clamp(depth, min=self.depth_range[0], max=self.depth_range[1])

			return depth

		def decode_location_flatten(self, points, offsets, depths, calibs, pad_size, batch_idxs):
			batch_size = len(calibs)
			gts = torch.unique(batch_idxs, sorted=True).tolist()
			locations = points.new_zeros(points.shape[0], 3).float()
			points = (points + offsets) * self.down_ratio - pad_size[batch_idxs]

			for idx, gt in enumerate(gts):
				corr_pts_idx = torch.nonzero(batch_idxs == gt).squeeze(-1)
				calib = calibs[gt]
				# concatenate uv with depth
				corr_pts_depth = torch.cat((points[corr_pts_idx], depths[corr_pts_idx, None]), dim=1)
				locations[corr_pts_idx] = calib.project_image_to_rect(corr_pts_depth)

			return locations

		def decode_y3d_from_bot_and_depth(self, points, pred_keypoints, pred_depths, calibs, pad_size, batch_idxs=None):
			batch_size = len(calibs)
			if batch_size == 1:
				batch_idxs = points.new_zeros(points.shape[0], dtype=torch.long)

			pred_bottom_points = pred_keypoints[:, -2, :]
			bottom_points_2d = (points + pred_bottom_points) * self.down_ratio - pad_size[batch_idxs]
			Vc = bottom_points_2d[:, 1]
			pred_y3d = []

			for idx, gt_idx in enumerate(torch.unique(batch_idxs, sorted=True).tolist()):
				calib = calibs[idx]
				b = calib.P[1, 3]
				c = calib.P[2, 3]
				corr_pts_idx = torch.nonzero(batch_idxs == gt_idx).squeeze(-1)
				y3d = (Vc[corr_pts_idx] - calib.c_v) * pred_depths[corr_pts_idx] / calib.f_v + (c * Vc[corr_pts_idx] - b) / calib.f_v
				pred_y3d.append(y3d)

			pred_y3d = torch.clamp(torch.cat(pred_y3d), min=self.y3d_range[0], max=self.y3d_range[1])
			return pred_y3d

		def decode_depth_from_roof_and_bottom(self, EL, center, pred_keypoints, pred_dimensions, calibs, pad_size, batch_idxs=None):
			batch_size = len(calibs)
			if batch_size == 1:
				batch_idxs = center.new_zeros(center.shape[0], dtype=torch.long)

			H = pred_dimensions[:, 1]
			if self.use_y3d_alone:
				center_point = (center[:, 1] + pred_keypoints[:, -2, 1]) * self.down_ratio - pad_size[batch_idxs][:, 1]
				corner_02_point = (center[:, 1].unsqueeze(1).repeat(1, 2) + pred_keypoints[:, [0, 2], 1]) * self.down_ratio - pad_size[batch_idxs][:, 1].unsqueeze(1).repeat(1, 2)
				corner_13_point = (center[:, 1].unsqueeze(1).repeat(1, 2) + pred_keypoints[:, [1, 3], 1]) * self.down_ratio - pad_size[batch_idxs][:, 1].unsqueeze(1).repeat(1, 2)
			elif self.compute_depth_only_from_roof:
				center_point = (center[:, 1] + pred_keypoints[:, -1, 1]) * self.down_ratio - pad_size[batch_idxs][:, 1]
				corner_02_point = (center[:, 1].unsqueeze(1).repeat(1, 2) + pred_keypoints[:, [4, 6], 1]) * self.down_ratio - pad_size[batch_idxs][:, 1].unsqueeze(1).repeat(1, 2)
				corner_13_point = (center[:, 1].unsqueeze(1).repeat(1, 2) + pred_keypoints[:, [5, 7], 1]) * self.down_ratio - pad_size[batch_idxs][:, 1].unsqueeze(1).repeat(1, 2)
			else:
				center_point = (center[:, 1] + (pred_keypoints[:, -2, 1] + pred_keypoints[:, -1, 1]) / 2) * self.down_ratio - pad_size[batch_idxs][:, 1]
				corner_02_point = (center[:, 1].unsqueeze(1).repeat(1, 2) + (pred_keypoints[:, [0, 2], 1] + pred_keypoints[:, [4, 6], 1]) / 2) * self.down_ratio - pad_size[batch_idxs][:, 1].unsqueeze(1).repeat(1, 2)
				corner_13_point = (center[:, 1].unsqueeze(1).repeat(1, 2) + (pred_keypoints[:, [1, 3], 1] + pred_keypoints[:, [5, 7], 1]) / 2) * self.down_ratio - pad_size[batch_idxs][:, 1].unsqueeze(1).repeat(1, 2)
			pred_compensated_depths = {'center': [], 'corner_02': [], 'corner_13': []}

			for idx, gt_idx in enumerate(torch.unique(batch_idxs, sorted=True).tolist()):
				calib = calibs[idx]
				b = calib.P[1, 3]
				c = calib.P[2, 3]
				corr_pts_idx = torch.nonzero(batch_idxs == gt_idx).squeeze(-1)
				if self.use_y3d_alone:
					center_depth = (EL[corr_pts_idx] * calib.f_v + b - center_point[corr_pts_idx] * c) / (center_point[corr_pts_idx] - calib.c_v)
					corner_02_depth = (EL[corr_pts_idx].unsqueeze(-1) * calib.f_v + b - corner_02_point[corr_pts_idx] * c) / (corner_02_point[corr_pts_idx] - calib.c_v)
					corner_13_depth = (EL[corr_pts_idx].unsqueeze(-1) * calib.f_v + b - corner_13_point[corr_pts_idx] * c) / (corner_13_point[corr_pts_idx] - calib.c_v)
				elif self.compute_depth_only_from_roof:
					center_depth = ((EL[corr_pts_idx] - H[corr_pts_idx]) * calib.f_v + b - center_point[corr_pts_idx] * c) / (center_point[corr_pts_idx] - calib.c_v)
					corner_02_depth = ((EL[corr_pts_idx] - H[corr_pts_idx]).unsqueeze(-1) * calib.f_v + b - corner_02_point[corr_pts_idx] * c) / (corner_02_point[corr_pts_idx] - calib.c_v)
					corner_13_depth = ((EL[corr_pts_idx] - H[corr_pts_idx]).unsqueeze(-1) * calib.f_v + b - corner_13_point[corr_pts_idx] * c) / (corner_13_point[corr_pts_idx] - calib.c_v)
				else:
					center_depth = ((EL[corr_pts_idx] - H[corr_pts_idx] / 2) * calib.f_v + b - center_point[corr_pts_idx] * c) / (center_point[corr_pts_idx] - calib.c_v)
					corner_02_depth = ((EL[corr_pts_idx] - H[corr_pts_idx] / 2).unsqueeze(-1) * calib.f_v + b - corner_02_point[corr_pts_idx] * c) / (corner_02_point[corr_pts_idx] - calib.c_v)
					corner_13_depth = ((EL[corr_pts_idx] - H[corr_pts_idx] / 2).unsqueeze(-1) * calib.f_v + b - corner_13_point[corr_pts_idx] * c) / (corner_13_point[corr_pts_idx] - calib.c_v)

				corner_02_depth = corner_02_depth.mean(dim=1)
				corner_13_depth = corner_13_depth.mean(dim=1)

				pred_compensated_depths['center'].append(center_depth)
				pred_compensated_depths['corner_02'].append(corner_02_depth)
				pred_compensated_depths['corner_13'].append(corner_13_depth)

			for key, depths in pred_compensated_depths.items():
				pred_compensated_depths[key] = torch.clamp(torch.cat(depths), min=self.depth_range[0], max=self.depth_range[1])

			pred_depths = torch.stack([depth for depth in pred_compensated_depths.values()], dim=1)

			return pred_depths

		def decode_depth_from_roof_and_bottom_multi_y3d(self, EL, center, pred_keypoints, pred_dimensions, calibs, pad_size, batch_idxs=None):
			batch_size = len(calibs)
			if batch_size == 1:
				batch_idxs = center.new_zeros(center.shape[0], dtype=torch.long)

			H = pred_dimensions[:, 1]
			if self.use_y3d_alone:
				center_point = (center[:, 1] + pred_keypoints[:, -2, 1]) * self.down_ratio - pad_size[batch_idxs][:, 1]
				corner_02_point = (center[:, 1].unsqueeze(1).repeat(1, 2) + pred_keypoints[:, [0, 2], 1]) * self.down_ratio - pad_size[batch_idxs][:, 1].unsqueeze(1).repeat(1, 2)
				corner_13_point = (center[:, 1].unsqueeze(1).repeat(1, 2) + pred_keypoints[:, [1, 3], 1]) * self.down_ratio - pad_size[batch_idxs][:, 1].unsqueeze(1).repeat(1, 2)
			elif self.compute_depth_only_from_roof:
				center_point = (center[:, 1] + pred_keypoints[:, -1, 1]) * self.down_ratio - pad_size[batch_idxs][:, 1]
				corner_02_point = (center[:, 1].unsqueeze(1).repeat(1, 2) + pred_keypoints[:, [4, 6], 1]) * self.down_ratio - pad_size[batch_idxs][:, 1].unsqueeze(1).repeat(1, 2)
				corner_13_point = (center[:, 1].unsqueeze(1).repeat(1, 2) + pred_keypoints[:, [5, 7], 1]) * self.down_ratio - pad_size[batch_idxs][:, 1].unsqueeze(1).repeat(1, 2)
			else:
				center_point = (center[:, 1] + (pred_keypoints[:, -2, 1] + pred_keypoints[:, -1, 1]) / 2) * self.down_ratio - pad_size[batch_idxs][:, 1]
				corner_02_point = (center[:, 1].unsqueeze(1).repeat(1, 2) + (pred_keypoints[:, [0, 2], 1] + pred_keypoints[:, [4, 6], 1]) / 2) * self.down_ratio - pad_size[batch_idxs][:, 1].unsqueeze(1).repeat(1, 2)
				corner_13_point = (center[:, 1].unsqueeze(1).repeat(1, 2) + (pred_keypoints[:, [1, 3], 1] + pred_keypoints[:, [5, 7], 1]) / 2) * self.down_ratio - pad_size[batch_idxs][:, 1].unsqueeze(1).repeat(1, 2)
			pred_compensated_depths = {'center': [], 'corner_02': [], 'corner_13': []}

			for idx, gt_idx in enumerate(torch.unique(batch_idxs, sorted=True).tolist()):
				calib = calibs[idx]
				b = calib.P[1, 3]
				c = calib.P[2, 3]
				corr_pts_idx = torch.nonzero(batch_idxs == gt_idx).squeeze(-1)
				if self.use_y3d_alone:
					center_depth = (EL[corr_pts_idx][:, -1] * calib.f_v + b - center_point[corr_pts_idx] * c) / (center_point[corr_pts_idx] - calib.c_v)
					corner_02_depth = (EL[corr_pts_idx][:, [0, 2]] * calib.f_v + b - corner_02_point[corr_pts_idx] * c) / (corner_02_point[corr_pts_idx] - calib.c_v)
					corner_13_depth = (EL[corr_pts_idx][:, [1, 3]] * calib.f_v + b - corner_13_point[corr_pts_idx] * c) / (corner_13_point[corr_pts_idx] - calib.c_v)
				elif self.compute_depth_only_from_roof:
					center_depth = ((EL[corr_pts_idx][:, -1] - H[corr_pts_idx]) * calib.f_v + b - center_point[corr_pts_idx] * c) / (center_point[corr_pts_idx] - calib.c_v)
					corner_02_depth = ((EL[corr_pts_idx][:, [0, 2]] - H[corr_pts_idx].unsqueeze(-1)) * calib.f_v + b - corner_02_point[corr_pts_idx] * c) / (corner_02_point[corr_pts_idx] - calib.c_v)
					corner_13_depth = ((EL[corr_pts_idx][:, [1, 3]] - H[corr_pts_idx].unsqueeze(-1)) * calib.f_v + b - corner_13_point[corr_pts_idx] * c) / (corner_13_point[corr_pts_idx] - calib.c_v)
				else:
					center_depth = ((EL[corr_pts_idx][:, -1] - H[corr_pts_idx] / 2) * calib.f_v + b - center_point[corr_pts_idx] * c) / (center_point[corr_pts_idx] - calib.c_v)
					corner_02_depth = ((EL[corr_pts_idx][:, [0, 2]] - H[corr_pts_idx].unsqueeze(-1) / 2) * calib.f_v + b - corner_02_point[corr_pts_idx] * c) / (corner_02_point[corr_pts_idx] - calib.c_v)
					corner_13_depth = ((EL[corr_pts_idx][:, [1, 3]] - H[corr_pts_idx].unsqueeze(-1) / 2) * calib.f_v + b - corner_13_point[corr_pts_idx] * c) / (corner_13_point[corr_pts_idx] - calib.c_v)

				corner_02_depth = corner_02_depth.mean(dim=1)
				corner_13_depth = corner_13_depth.mean(dim=1)

				pred_compensated_depths['center'].append(center_depth)
				pred_compensated_depths['corner_02'].append(corner_02_depth)
				pred_compensated_depths['corner_13'].append(corner_13_depth)

			for key, depths in pred_compensated_depths.items():
				pred_compensated_depths[key] = torch.clamp(torch.cat(depths), min=self.depth_range[0], max=self.depth_range[1])

			pred_depths = torch.stack([depth for depth in pred_compensated_depths.values()], dim=1)

			return pred_depths

		def decode_depth_from_keypoints(self, pred_offsets, pred_keypoints, pred_dimensions, calibs, avg_center=False):
			# pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center
			assert len(calibs) == 1 # for inference, batch size is always 1
			
			calib = calibs[0]
			# we only need the values of y
			pred_height_3D = pred_dimensions[:, 1]
			pred_keypoints = pred_keypoints.view(-1, 10, 2)
			# center height -> depth
			if avg_center:
				updated_pred_keypoints = pred_keypoints - pred_offsets.view(-1, 1, 2)
				center_height = updated_pred_keypoints[:, -2:, 1]
				center_depth = calib.f_u * pred_height_3D.unsqueeze(-1) / (center_height.abs() * self.down_ratio * 2)
				center_depth = center_depth.mean(dim=1)
			else:
				center_height = pred_keypoints[:, -2, 1] - pred_keypoints[:, -1, 1]
				center_depth = calib.f_u * pred_height_3D / (center_height.abs() * self.down_ratio)
			
			# corner height -> depth
			corner_02_height = pred_keypoints[:, [0, 2], 1] - pred_keypoints[:, [4, 6], 1]
			corner_13_height = pred_keypoints[:, [1, 3], 1] - pred_keypoints[:, [5, 7], 1]
			corner_02_depth = calib.f_u * pred_height_3D.unsqueeze(-1) / (corner_02_height * self.down_ratio)
			corner_13_depth = calib.f_u * pred_height_3D.unsqueeze(-1) / (corner_13_height * self.down_ratio)
			corner_02_depth = corner_02_depth.mean(dim=1)
			corner_13_depth = corner_13_depth.mean(dim=1)
			# K x 3
			pred_depths = torch.stack((center_depth, corner_02_depth, corner_13_depth), dim=1)

			return pred_depths

		def decode_depth_from_keypoints_batch(self, pred_keypoints, pred_dimensions, calibs, batch_idxs=None):
			# pred_keypoints: K x 10 x 2, 8 vertices, bottom center and top center
			pred_height_3D = pred_dimensions[:, 1].clone()
			batch_size = len(calibs)
			if batch_size == 1:
				batch_idxs = pred_dimensions.new_zeros(pred_dimensions.shape[0])

			center_height = pred_keypoints[:, -2, 1] - pred_keypoints[:, -1, 1]
			corner_02_height = pred_keypoints[:, [0, 2], 1] - pred_keypoints[:, [4, 6], 1]
			corner_13_height = pred_keypoints[:, [1, 3], 1] - pred_keypoints[:, [5, 7], 1]

			pred_keypoint_depths = {'center': [], 'corner_02': [], 'corner_13': []}

			for idx, gt_idx in enumerate(torch.unique(batch_idxs, sorted=True).tolist()):			
				calib = calibs[idx]
				corr_pts_idx = torch.nonzero(batch_idxs == gt_idx).squeeze(-1)
				center_depth = calib.f_u * pred_height_3D[corr_pts_idx] / (F.relu(center_height[corr_pts_idx]) * self.down_ratio + self.EPS)
				corner_02_depth = calib.f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(corner_02_height[corr_pts_idx]) * self.down_ratio + self.EPS)
				corner_13_depth = calib.f_u * pred_height_3D[corr_pts_idx].unsqueeze(-1) / (F.relu(corner_13_height[corr_pts_idx]) * self.down_ratio + self.EPS)

				corner_02_depth = corner_02_depth.mean(dim=1)
				corner_13_depth = corner_13_depth.mean(dim=1)

				pred_keypoint_depths['center'].append(center_depth)
				pred_keypoint_depths['corner_02'].append(corner_02_depth)
				pred_keypoint_depths['corner_13'].append(corner_13_depth)

			for key, depths in pred_keypoint_depths.items():
				pred_keypoint_depths[key] = torch.clamp(torch.cat(depths), min=self.depth_range[0], max=self.depth_range[1])

			pred_depths = torch.stack([depth for depth in pred_keypoint_depths.values()], dim=1)

			return pred_depths

		def decode_dimension(self, cls_id, dims_offset):
			'''
			retrieve object dimensions
			Args:
					cls_id: each object id
					dims_offset: dimension offsets, shape = (N, 3)

			Returns:

			'''
			cls_id = cls_id.flatten().long()
			cls_dimension_mean = self.dim_mean[cls_id, :]
			# dim_modes is ['exp', True, False]
			if self.dim_modes[0] == 'exp':
				dims_offset = dims_offset.exp()

			if self.dim_modes[2]:
				cls_dimension_std = self.dim_std[cls_id, :]
				dimensions = dims_offset * cls_dimension_std + cls_dimension_mean
			else:
				dimensions = dims_offset * cls_dimension_mean
				
			return dimensions

		def decode_y3d_from_ground_plane(self, equs, points, pred_keypoints, calibs, pad_size, batch_idxs=None):
			batch_size = len(calibs)
			if batch_size == 1:
				batch_idxs = points.new_zeros(points.shape[0], dtype=torch.long)

			bot_2D = (points + pred_keypoints[:, 8]) * self.down_ratio - pad_size[batch_idxs]
			U = bot_2D[:, 0]
			V = bot_2D[:, 1]
			pred_y3d = []

			for idx, gt_idx in enumerate(torch.unique(batch_idxs, sorted=True).tolist()):
				calib = calibs[idx]
				equ = equs[idx]
				a, b, c, d = equ[0], equ[1], equ[2], equ[3]
				corr_pts_idx = torch.nonzero(batch_idxs == gt_idx).squeeze(-1)
				n = (calib.f_v / calib.f_u) * (U[corr_pts_idx] - calib.c_u) / (V[corr_pts_idx] - calib.c_v)
				m = calib.f_v / (V[corr_pts_idx] - calib.c_v)
				y3d = - (d / (a*n + c*m + b))
				pred_y3d.append(y3d)

			pred_y3d = torch.clamp(torch.cat(pred_y3d), min=self.y3d_range[0], max=self.y3d_range[1])

			return pred_y3d

		def decode_multi_y3d_from_ground_plane(self, equs, points, pred_keypoints, calibs, pad_size, batch_idxs=None):
			batch_size = len(calibs)
			if batch_size == 1:
				batch_idxs = points.new_zeros(points.shape[0], dtype=torch.long)

			bot_2D = (points.unsqueeze(1).repeat(1, 5, 1) + pred_keypoints[:, [0, 1, 2, 3, 8]]) * self.down_ratio - \
					 pad_size[batch_idxs].unsqueeze(1).repeat(1, 5, 1)
			U = bot_2D[:, :, 0]
			V = bot_2D[:, :, 1]
			pred_y3d = {'0': [], '1': [], '2': [], '3': [], '4': []}

			for idx, gt_idx in enumerate(torch.unique(batch_idxs, sorted=True).tolist()):
				calib = calibs[idx]
				equ = equs[idx]
				a, b, c, d = equ[0], equ[1], equ[2], equ[3]
				corr_pts_idx = torch.nonzero(batch_idxs == gt_idx).squeeze(-1)
				n = (calib.f_v / calib.f_u) * (U[corr_pts_idx] - calib.c_u) / (V[corr_pts_idx] - calib.c_v)
				m = calib.f_v / (V[corr_pts_idx] - calib.c_v)
				y3d = - (d / (a * n + c * m + b))
				for i in range(5):
					pred_y3d[str(i)].append(y3d[:, i])

			for key, y3d in pred_y3d.items():
				pred_y3d[key] = torch.clamp(torch.cat(y3d), min=self.y3d_range[0], max=self.y3d_range[1])

			pred_y3d = torch.stack([y3d for y3d in pred_y3d.values()], dim=1)

			return pred_y3d

		def decode_y3d(self, cls_id, y3d_offset):
			cls_id = cls_id.flatten().long()
			y3d_offset = y3d_offset.flatten()
			cls_y3d_mean = self.y3d_mean[cls_id]


			if self.y3d_modes[0] == 'exp':
				y3d_offset = y3d_offset.exp()
			elif self.y3d_modes[0] == 'log':
				y3d_offset = y3d_offset.log()

			if self.y3d_modes[1]:
				cls_y3d_std = self.y3d_std[cls_id]
				y3d = y3d_offset * cls_y3d_std + cls_y3d_mean
			else:
				y3d = y3d_offset * cls_y3d_mean

			return y3d


		def decode_axes_orientation(self, vector_ori, locations):
			'''
			retrieve object orientation
			Args:
					vector_ori: local orientation in [axis_cls, head_cls, sin, cos] format
					locations: object location

			Returns: for training we only need roty
							 for testing we need both alpha and roty

			'''
			if self.multibin:
				pred_bin_cls = vector_ori[:, : self.orien_bin_size * 2].view(-1, self.orien_bin_size, 2)
				pred_bin_cls = torch.softmax(pred_bin_cls, dim=2)[..., 1]
				orientations = vector_ori.new_zeros(vector_ori.shape[0])
				for i in range(self.orien_bin_size):
					mask_i = (pred_bin_cls.argmax(dim=1) == i)
					s = self.orien_bin_size * 2 + i * 2
					e = s + 2
					pred_bin_offset = vector_ori[mask_i, s : e]
					orientations[mask_i] = torch.atan2(pred_bin_offset[:, 0], pred_bin_offset[:, 1]) + self.alpha_centers[i]
			else:
				axis_cls = torch.softmax(vector_ori[:, :2], dim=1)
				axis_cls = axis_cls[:, 0] < axis_cls[:, 1]
				head_cls = torch.softmax(vector_ori[:, 2:4], dim=1)
				head_cls = head_cls[:, 0] < head_cls[:, 1]
				# cls axis
				orientations = self.alpha_centers[axis_cls + head_cls * 2]
				sin_cos_offset = F.normalize(vector_ori[:, 4:])
				orientations += torch.atan(sin_cos_offset[:, 0] / sin_cos_offset[:, 1])

			locations = locations.view(-1, 3)
			rays = torch.atan2(locations[:, 0], locations[:, 2])
			alphas = orientations
			rotys = alphas + rays

			larger_idx = (rotys > PI).nonzero()
			small_idx = (rotys < -PI).nonzero()
			if len(larger_idx) != 0:
					rotys[larger_idx] -= 2 * PI
			if len(small_idx) != 0:
					rotys[small_idx] += 2 * PI

			larger_idx = (alphas > PI).nonzero()
			small_idx = (alphas < -PI).nonzero()
			if len(larger_idx) != 0:
					alphas[larger_idx] -= 2 * PI
			if len(small_idx) != 0:
					alphas[small_idx] += 2 * PI

			return rotys, alphas

		def decode_ground_plane_from_heatmap(self, pred_horizon_heatmaps, horizon_states, calibs, pad_sizes):
			pred_ground_planes = []
			batch = len(calibs)
			for i in range(batch):
				calib = calibs[i]
				horizon_state = horizon_states[i]
				pad_size = pad_sizes[i].cpu().numpy()
				pad_x = int(np.ceil(pad_size[0] / 4))
				horizon_heatmap = pred_horizon_heatmaps[i][0]
				# find the index of the maximum value of each column from horizon_heatmap
				v = torch.argmax(horizon_heatmap[:, pad_x:horizon_heatmap.shape[1]-pad_x], dim=0)
				u = torch.arange(pad_x, horizon_heatmap.shape[1]-pad_x, device='cuda')
				if self.horizon_fitting_method == 'LS':
					points = torch.cat((u.unsqueeze(1), v.unsqueeze(1)), dim=1)
					points = points.cpu().numpy()
					[vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.1, 0.01)
					K = float(vy) / float(vx)  # Line slope
					B = (-K * x + y)[0]
				elif self.horizon_fitting_method == 'RANSAC':
					ransac = linear_model.RANSACRegressor()
					ransac.fit(u.reshape(-1, 1), v)
					K = ransac.estimator_.coef_[0]
					B = ransac.estimator_.intercept_

				Kh = K
				bh = 4 * B + K * pad_size[0] - pad_size[1]

				if self.use_edge_slope:
					if horizon_state[0]:
						Kh = horizon_state[1]

				f_x, f_y, c_x, c_y = calib.f_u, calib.f_v, calib.c_u, calib.c_v
				m = Kh * f_x / f_y
				n = (Kh * c_x + bh - c_y) / f_y
				formula = (1 / (1 + m**2 + n**2))**0.5
				a = m * formula
				b = -formula
				c = n * formula
				d = 1.65
				pred_ground_plane = torch.tensor([a, b, c, d], device='cuda')
				pred_ground_planes.append(pred_ground_plane)

			return torch.stack(pred_ground_planes, dim=0)

		def decode_original_box2d_batch(self, box2d, trans_inv):
			trans_inv = trans_inv[0].float()

			# box2d n×4
			# transform box2d to the original image
			for i in range(box2d.size()[0]):
				boxi = box2d[i].reshape(2, 2)
				boxi = boxi.transpose(0, 1)
				boxi = torch.cat((boxi, torch.ones(1, 2, device='cuda')), dim=0)
				new_box = trans_inv @ boxi
				new_box = new_box.transpose(0, 1)
				box2d[i] = new_box[:, :2].reshape(-1)

			return box2d

		def decode_depth_with_mode(self, depths_offset, mode):
			if mode == 'exp':
				depth = depths_offset.exp()
			elif mode == 'linear':
				depth = depths_offset * self.depth_ref[1] + self.depth_ref[0]
			elif mode == 'inv_sigmoid':
				depth = 1 / torch.sigmoid(depths_offset) - 1
			else:
				raise ValueError

			if self.depth_range is not None:
				depth = torch.clamp(depth, min=self.depth_range[0], max=self.depth_range[1])

			return depth


if __name__ == '__main__':
	pass