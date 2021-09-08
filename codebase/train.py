from loss_function import *
from depth_model import *
from Dataloader import Kittiloader
from dataset import DataGenerator
from evaluate_model import evaluate
import argparse
import pathlib
import numpy as np
import time
import pickle
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image


parser = argparse.ArgumentParser(description='Training parameters.')

parser.add_argument('--save_every', type=int, default=1,
                    help='after n epochs our model will save')
parser.add_argument('--save_file', type=pathlib.Path,
                    help='save path for the model')
parser.add_argument('--results_file', type=pathlib.Path,
                    help='save path for model results')
parser.add_argument('--kitti_dir', type=pathlib.Path)
parser.add_argument('--epochs', type=int, default=1,
                    help='how many epochs to run for')

args = parser.parse_args()


def generate_image(image, disparity):
	scale = disparity.shape[2] / image.shape[2]
	scaled_img = scale_image(image, scale)
	return apply_disparity(scaled_img, disparity)


def scale_image(image, scale):
	return F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=True)


def apply_disparity(image, disp):
	b, _, h, w = image.size()
	# map location of original pixels
	y_mesh, x_mesh = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w))
	# add dimension(s) for batch size
	x_mesh = x_mesh.repeat(b, 1, 1).cuda()
	y_mesh = y_mesh.repeat(b, 1, 1).cuda()
	# Apply shift in X direction
	applied = x_mesh + disp[:,0,:,:]  # drop the "other" image from disparity (L-R)
	flow_field = torch.stack((applied, y_mesh), dim=3)  #
	flow_field = 2 * flow_field - 1  # recenter points from [0,1] to [-1,1] (needed for grid_sample)
	# apply shift to original image
	output = F.grid_sample(image, flow_field, mode='bilinear', padding_mode='zeros')
	return output


def train_one_epoch(model, opti, loss_model, dataloader):

	for id, batch in enumerate(tqdm(dataloader)):
		batch_losses = []
		left_img = batch['left_img'].cuda()
		right_img = batch['right_img'].cuda()
		disp1, disp2, disp3, disp4 = model(left_img)
		disp1l, disp1r = disp1[:, 0,:,:].unsqueeze(1), disp1[:, 1,:,:].unsqueeze(1)
		disp2l, disp2r = disp2[:, 0,:,:].unsqueeze(1), disp2[:, 1,:,:].unsqueeze(1)
		disp3l, disp3r = disp3[:, 0,:,:].unsqueeze(1), disp3[:, 1,:,:].unsqueeze(1)
		disp4l, disp4r = disp4[:, 0,:,:].unsqueeze(1), disp4[:, 1,:,:].unsqueeze(1)
		# def loss(self, generated_image, true_image, left_disparity, right_projected_disparity):
		loss1l2r = loss_model.loss(generate_image(left_img, disp1r), scale_image(right_img, 0.5 ** 0), disp1r, generate_image(disp1l, disp1r), 1)
		loss1r2l = loss_model.loss(generate_image(right_img, -disp1l), scale_image(left_img, 0.5 ** 0), disp1l, generate_image(disp1r, -disp1l), 1)

		loss2l2r = loss_model.loss(generate_image(left_img, disp2r), scale_image(right_img, 0.5 ** 1), disp2r, generate_image(disp2l, disp2r), 2)
		loss2r2l = loss_model.loss(generate_image(right_img, -disp2l), scale_image(left_img, 0.5 ** 1), disp2l, generate_image(disp2r, -disp2l), 2)

		loss3l2r = loss_model.loss(generate_image(left_img, disp3r), scale_image(right_img, 0.5 ** 2), disp3r, generate_image(disp3l, disp3r), 3)
		loss3r2l = loss_model.loss(generate_image(right_img, -disp3l), scale_image(left_img, 0.5 ** 2), disp3l, generate_image(disp3r, -disp3l), 3)

		loss4l2r = loss_model.loss(generate_image(left_img, disp4r), scale_image(right_img, 0.5 ** 3), disp4r, generate_image(disp4l, disp4r), 4)
		loss4r2l = loss_model.loss(generate_image(right_img, -disp4l), scale_image(left_img, 0.5 ** 3), disp4l, generate_image(disp4r, -disp4l), 4)


		loss = loss1l2r + loss1r2l + loss2l2r + loss2r2l + loss3l2r + loss3r2l + loss4l2r + loss4r2l
		opti.zero_grad()
		loss.backward()
		opti.step()
		batch_losses.append(loss.item())
	return batch_losses

def main():
	model = MonoDepthModel()
	model = model.cuda()
	loss = MonoLoss()

	global batch_size
	batch_size = 22
	epochs = range(args.epochs)
	#Optional for later, we can freeze certain layers
	opti = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.01)
	# lr_sched = torch.optim.lr_scheduler.StepLR(opti, step_size=50, gamma=0.1)

	# For testing
	# datagen= DataGenerator('kitti_data', phase='small', splits='kitti')
	# dataloader = datagen.create_data(batch_size)

	# For running
	datagen = DataGenerator(args.kitti_dir, phase='train', splits='kitti')
	datagen_val = DataGenerator(args.kitti_dir, phase='val', splits='kitti')
	datagen_test = DataGenerator(args.kitti_dir, phase='test', splits='eigen')
	dataloader = datagen.create_data(batch_size)
	val_dataloader = datagen_val.create_data(batch_size)
	test_dataloader = datagen_test.create_data(batch_size)

	global start
	start = time.time()
	results = {"train": {}}
	for epoch in epochs:
		epoch_start = time.time()
		batch_losses = train_one_epoch(model, opti, loss, dataloader)
		# lr_sched.step()
		print(f"Mean Batch Loss for Epoch {epoch} was {np.mean(batch_losses)}")
		results['train'][epoch] = batch_losses

		torch.save(model, f'{args.save_file}/monodepth_epoch_{epoch}')


		# save model checkpoints
		if not epoch & args.save_every:
			torch.save(model, f'{args.save_file}/monodepth_epoch_{epoch}')

		epoch_end = time.time()
		print(f"Finished epoch {epoch} in {(epoch_end - epoch_start) / 60:.5} minutes")

	# save final results
	with open(f"{args.results_file}", "w") as f:
		pickle.dump(results, f)

if __name__ == '__main__':
	main()
