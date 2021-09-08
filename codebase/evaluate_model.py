import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

MIN_DEPTH = 1e-3
MAX_DEPTH = 80
SCALE_FACTOR = 5.4/17.7

def get_depth_map(disp, H, W):
	resized_disp = F.interpolate(disp.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear").squeeze()

	depth_est = SCALE_FACTOR / resized_disp

	return depth_est
def crop(depth_map):
	# We copied the crops floats for evaluation from the origonal implementation.
	# These crops are somewhat arbitrary, so coping them is the only way
	# to get a good evaluation metric.

	h, w = depth_map.shape
	return depth_map[int(0.3924324 * h):int(0.91351351 * h), int(0.0359477 * w):int(0.96405229 * w)]


def abs_rel(pred, target):
	return torch.mean(torch.abs(target - pred) / target)

def sq_rel(pred, target):
	return torch.mean(torch.abs(target - pred) ** 2 / target)

def rmse(pred, target):
	return torch.mean(torch.abs(target - pred) ** 2) ** 0.5

def rmse_log(pred, target):
	return torch.mean(torch.abs(torch.log(target) - torch.log(pred))) ** 0.5

def deltas(pred, target):
	delta = torch.max(pred/target, target/pred)
	metric1 = torch.mean((delta < 1.25).float())
	metric2 = torch.mean((delta < 1.25 ** 2).float())
	metric3 = torch.mean((delta < 1.25 ** 3).float())
	return metric1, metric2, metric3

def all_metrics(pred, target):
	tmp = abs_rel(pred, target), sq_rel(pred, target), rmse(pred, target), rmse_log(pred, target), *deltas(pred, target)
	return list(map(lambda x: x.item(), tmp))

# def post_process_disparity(disp1l, model, left_img):
# 	flipped_disp = torch.flip(model(torch.flip(left_img, (2,)))[0][:, 0, :, :], (2,))
# 	b, c, h, w = disp1l.shape
# 	_, x = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w))
# 	ones_mask = (x <= 0.1).float()
# 	halfs_mask = (x > 0.1).float() * 0.5
# 	mask = ones_mask + halfs_mask
# 	flipped_mask = torch.flip(mask, (2,))
# 	return (disp1l * mask) + (flipped_disp * flipped_mask)

def evaluate(model, test_dataloader, *, display=False, post_process=False):
	batch_results = []
	for batch in tqdm(test_dataloader):
		left_img = batch['left_img'].cuda()
		depth_map_interpolated = batch['depth_interp'].cuda()
		#The focal lenght and baselines should be the same for
		#All images in a batch so taking the mean is just a reduction
		focal_length = torch.mean(batch['focal_length'].cuda())
		baseline = torch.mean(batch['baseline'].cuda())
		disp1, _, _, _ = model(left_img)
		# disp1l, disp1r = disp1[:, 0,:,:], disp1[:, 1,:,:]
		disp1l = disp1[:, 0, :, :].squeeze()
		# if post_process:
		# 	disp1l = post_process_disparity(disp1l, model, left_img)

		depth_map_interpolated = depth_map_interpolated.squeeze()

		h, w = depth_map_interpolated.shape

		pred_depth_map = get_depth_map(disp1l, h, w)

		cropped_pred = crop(pred_depth_map)
		cropped_truth = crop(depth_map_interpolated)

		cropped_truth[cropped_truth > MAX_DEPTH] = MAX_DEPTH
		cropped_truth[cropped_truth < MIN_DEPTH] = MIN_DEPTH

		# resize_depth_map = F.resize(depth_map_interpolated, [256, 512])
		ratio = torch.median(cropped_truth) / torch.median(cropped_pred)
		cropped_pred *= ratio
		# import pdb;
		# pdb.set_trace()


		cropped_pred[cropped_pred > MAX_DEPTH] = MAX_DEPTH
		cropped_pred[cropped_pred < MIN_DEPTH] = MIN_DEPTH

		batch_result = all_metrics(cropped_pred, cropped_truth)
		if display:
			print(batch_result)
		batch_results.append(batch_result)
	return np.median(batch_results, axis=0)

