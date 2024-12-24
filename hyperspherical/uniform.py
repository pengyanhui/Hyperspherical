import math

import torch
from torch.nn import functional as func
from torch import distributions
from torch.distributions import constraints


# 由此可见, 实现 Distribution 的部分方法就可以了
class HypersphericalUniform(distributions.Distribution):
	arg_constraints = {}
	support = constraints.real_vector

	def __init__(self, dim, batch_shape=torch.Size(), dtype=torch.float32, device='cpu', validate_args=None):
		self._dim = dim
		self._dtype = dtype
		self._device = device

		super().__init__(batch_shape, torch.Size([dim]), validate_args=validate_args)

	def sample(self, shape=torch.Size()):
		"""
		:param shape: 应该是 sample 排列的形状, 如 shape=[2,3], 指 2*3 个 sample, [] 指 1 个
		:return: 服从标准均匀分布的 dim 维单位向量
		"""
		x = torch.randn(
			torch.Size(shape + self.batch_shape + self.event_shape),
			dtype=self._dtype, device=self._device
		)
		func.normalize(x.data, dim=-1, out=x.data)
		return x

	def entropy(self):
		"""
		:return: batch_shape 个熵
		"""
		return torch.full(self.batch_shape, self._log_surface_area(), dtype=self._dtype, device=self._device)

	def log_prob(self, x: torch.Tensor):
		"""
		:param x: 张量, 最后一维是 dim 维单位向量;
		:return: 假设 x.shape = [4,5,dim], 返回 [4,5,*batch_shape] 个 log_prob
		"""
		if self._validate_args:
			self._validate_sample(x)
		if x.dim() == 1:
			bshape = torch.Size([])
		else:
			bshape = x.shape[:-len(self.batch_shape) - 1]
		return torch.full(bshape + self.batch_shape, -self._log_surface_area(), dtype=self._dtype, device=self._device)

	def _log_surface_area(self):
		# 超球面的面积: https://en.wikipedia.org/wiki/N-sphere
		# 对 scalar 而言, math.log, math.lgamma 竟然要快得多
		return math.log(2) + (self._dim / 2) * math.log(math.pi) - math.lgamma(self._dim / 2)


if __name__ == '__main__':
	uniform_sphere = HypersphericalUniform(8, batch_shape=torch.Size([3, 4]), device='cuda:0')
	samples = uniform_sphere.sample(torch.Size([2]))
	print(samples)
	# print(uniform_sphere.entropy())
	print(uniform_sphere.log_prob(samples))
	# print(uniform_sphere.log_prob(torch.randn(5)))
