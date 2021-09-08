import torch
import kornia


class MonoLoss(torch.nn.Module):

	def __init__(self,
				 gaussian_filter_size=(3, 3),
				 gaussian_filter_std=(1.5, 1.5),
				 C1=0.001,
				 C2=0.001,
				 alpha=0.85,
				 aml_w=1,
				 dsl_w=1,
				 lrdcl_w=0.1):
		super(MonoLoss, self).__init__()
		self.gaussian_filter = kornia.filters.GaussianBlur2d(gaussian_filter_size, gaussian_filter_std)
		self.C1 = C1
		self.C2 = C2
		self.alpha = alpha
		self.aml_w = aml_w
		self.dsl_w = dsl_w
		self.lrdcl_w = lrdcl_w

	def appearance_matching_loss(self, gen_img, true_img):
		# First we need to calculated the L1 Loss
		l1_loss = torch.mean(torch.abs(gen_img - true_img), dim=1).unsqueeze(1)

		# Second we need to calculate the SSIM Lossf
		"""
		This differs from the paper b/c we use the gaussian filter that was
		suggested in the paper the invented SSIM. The paper we are reproducing used 
		a block filter instead. I would expect that gaussian filter helps. 
		"""
		mu_x, mu_y = self.gaussian_filter(gen_img), self.gaussian_filter(true_img)
		sigma_x, sigma_y = self.gaussian_filter(gen_img ** 2) - (mu_x ** 2), self.gaussian_filter(true_img ** 2) - (mu_y ** 2)
		sigma_xy = self.gaussian_filter(gen_img * true_img) - (mu_x * mu_y)

		ssim = (((2 * mu_x * mu_y) + self.C1) / (mu_x ** 2 + mu_y ** 2 + self.C1)) * \
			((2 * sigma_xy + self.C2) / (sigma_x + sigma_y + self.C2))
		ssim_loss = 1 - torch.clamp(ssim, 0, 1)

		loss = torch.mean((self.alpha * ssim_loss / 2) + ((1 - self.alpha) * l1_loss))
		return loss

	@staticmethod
	def disparity_smoothness_loss(disparity, true_image):
		disparity_grad = torch.abs(kornia.spatial_gradient(disparity))
		d_x, d_y = disparity_grad[:, :, 0, :, :], disparity_grad[:, :, 1, :, :]

		image_grad = kornia.spatial_gradient(true_image)
		i_x, i_y = image_grad[:, :, 0, :, :], image_grad[:, :, 1, :, :]

		e_pwr_i_x, e_pwr_i_y = torch.exp(-torch.norm(i_x, dim=1).unsqueeze(1)), torch.exp(
			-torch.norm(i_y, dim=1).unsqueeze(1))

		loss = torch.mean(d_x * e_pwr_i_x + d_y * e_pwr_i_y)
		return loss

	@staticmethod
	def left_right_disparity_consistency_loss(l_disparity, r_proj_disparity):
		return torch.mean(torch.abs(l_disparity - r_proj_disparity))

	def loss(self, generated_image, true_image, left_disparity, right_projected_disparity, scale):
		aml = self.appearance_matching_loss(generated_image, true_image)
		dsl = self.disparity_smoothness_loss(left_disparity, true_image)
		lrdcl = self.left_right_disparity_consistency_loss(left_disparity, right_projected_disparity)

		return self.aml_w * aml + self.dsl_w * dsl + (self.lrdcl_w / scale) * lrdcl
