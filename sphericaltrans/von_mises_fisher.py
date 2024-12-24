import math
from numbers import Number
from typing import Union

import torch
from torch import distributions
from torch.distributions import constraints
from torch.nn import functional as func

from sphericaltrans.bessel import ive
from sphericaltrans.jointtv import JointTSDistribution
from sphericaltrans.uniform import HypersphericalUniform
from sphericaltrans.transforms import TTransform, HouseholderRotationTransform


class _MarginalTDistribution(distributions.Distribution):
	arg_constraints = {'kappa': constraints.positive}
	has_rsample = True
	_EPS = 1e-36

	def __init__(self, dim, kappa, dtype=None, validate_args=None):
		self.dim = dim
		self.kappa = kappa
		super().__init__(batch_shape=kappa.shape, validate_args=validate_args)

		self._dtype = dtype if dtype is not None else kappa.dtype
		self._device = kappa.device

		# >>> for sampling algorithm >>>
		self._uniform = distributions.Uniform(self._EPS, torch.tensor(1.0 - self._EPS, dtype=self._dtype, device=self._device))
		self._beta = distributions.Beta(
			torch.tensor((self.dim - 1) / 2, dtype=self._dtype, device=self._device),
			torch.tensor((self.dim - 1) / 2, dtype=self._dtype, device=self._device)
		)

	def sample(self, shape: Union[torch.Size, int] = torch.Size()):
		if isinstance(shape, int):
			shape = torch.Size([shape])
		with torch.no_grad():  # rsample 是 reparameterized sample, 便于梯度更新以调整分布参数
			return self.rsample(shape)

	def rsample(self, shape=torch.Size()):
		"""
		Reparameterized Sample: 从一个简单的分布通过一个参数化变换使得其满足一个更复杂的分布;
		此处, mu 是可变参数, 通过 radial-tangential decomposition 采样;
		梯度下降更新 mu, 以获得满足要求的 vMF.
		:param shape: 样本的形状
		:return: [shape|m] 的张量, shape 个 m 维方向向量
		"""
		# shape = torch.Size(shape + self._batch_shape + self._event_shape)
		w = (
			self._sample_w3(shape=shape)
			if self.dim == 3
			else self._sample_w_rej(shape=shape)
		)
		return w

	def _sample_w3(self, shape: torch.Size):
		"""
		拒绝采样方法来自: Computer Generation of Distributions on the M-Sphere
		https://rss.onlinelibrary.wiley.com/doi/abs/10.2307/2347441
		:param shape: 采样 w 的的形状
		:return: 形状为 shape 的张量, shape 个 w
		"""
		# kappa 也是有 shape 的, 说明可以并行多个 κ 吗?
		shape = torch.Size(shape + self.kappa.shape)  # torch.Size 继承自 tuple, 其 + 运算就是连接操作
		# https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution # 3-D sphere
		u = self._uniform.sample(shape)
		w = 1 + torch.stack(  # 这个公式是按 μ=(0,0,1) 计算的 w, arccosw=φ, 即 w=z
			[  # 最后的旋转可能是旋转至按真正的 μ 采样结果
				torch.log(u),
				torch.log(1 - u) - 2 * self.kappa
			],
			dim=0
		).logsumexp(0) / self.kappa
		return w

	def _sample_w_rej(self, shape: torch.Size):
		num_need = math.prod(shape)
		num_kappa = math.prod(self.kappa.shape)
		sample_shape = torch.Size([num_kappa, 10 + math.ceil(num_need * 1.3)])

		kappa = self.kappa.reshape(-1, 1)
		c = torch.sqrt(4 * kappa.square() + (self.dim - 1) ** 2)
		b = (-2 * kappa + c) / (self.dim - 1)
		t_0 = (-(self.dim - 1) + c) / (2 * kappa)
		s = kappa * t_0 + (self.dim - 1) * torch.log(1 - t_0.square())

		# 大天坑, [[]] * num_kappa 虽然也形成了 [num_kappa]个[]], 但实际上这些 [] 是同一个对象;
		# 导致后面的 ts[i] 全是同一个 [], 并 append 了所有不同 kappa 对应的 t 样本;
		# [torch.cat(t)[:num_need] for t in ts] 中每个 tensor 也仅仅获取了第一个 kappa 对应的 t 样本;
		# 因为 Python 的 * 运算符是浅拷贝, 而不是深拷贝.
		ts = [[] for _ in range(num_kappa)]  # * num_kappa
		cnts = torch.zeros(num_kappa, device=self._device)
		while cnts.lt(num_need).any():
			y = self._beta.sample(sample_shape)
			u = self._uniform.sample(sample_shape)
			t = (1 - (1 + b) * y) / (1 - (1 - b) * y)
			mask = (kappa * t + (self.dim - 1) * torch.log(1 - t_0 * t) - s) > torch.log(u)
			for i in range(num_kappa):
				ts[i].append(t[i][mask[i]])
			cnts += mask.sum(dim=-1)
		samples = torch.stack([torch.cat(t)[:num_need] for t in ts], dim=1)
		return samples.reshape(shape + self.kappa.shape)

	def log_prob(self, value: torch.Tensor) -> torch.Tensor:
		result = value * self.kappa + (self.dim / 2 - 1 - 0.5) * torch.log(1 - value.square())
		result += (self.dim / 2 - 1) * torch.log(self.kappa / 2)
		result -= math.lgamma(0.5) + math.lgamma(self.dim / 2 - 0.5) + torch.log(ive(self.dim / 2 - 1, self.kappa)) + self.kappa
		return result


class VonMisesFisher(distributions.TransformedDistribution):
	arg_constraints = {
		'mu': constraints.real_vector,
		'kappa': constraints.positive
	}
	has_rsample = True

	def __init__(self, mu, kappa, validate_args=None):
		"""
		:param mu: μ 待优化
		:param kappa: kappa
		:param validate_args: 是否检查类型限制
		"""
		assert mu.dim() >= 1 and mu.shape[-1] >= 3, '方向维度至少为 3, 二维情况直接用 distributions.VonMises'
		func.normalize(mu.data, dim=-1, out=mu.data)
		if isinstance(kappa, Number):
			kappa = torch.tensor(kappa, dtype=mu.dtype, device=mu.device)

		self.mu, self.kappa = mu, kappa
		super().__init__(
			JointTSDistribution(
				_MarginalTDistribution(
					mu.shape[-1], kappa, validate_args=validate_args
				),
				HypersphericalUniform(
					mu.shape[-1] - 1,
					batch_shape=mu.shape[:-1],
					device=mu.device, dtype=mu.dtype, validate_args=validate_args
				)
			),
			[TTransform(), HouseholderRotationTransform(mu)]
		)

	def log_prob_true(self, x):
		if self._validate_args:
			self._validate_sample(x)
		return self._log_unnormalized_prob(x) + self._log_normalization()

	def _log_unnormalized_prob(self, x):  # k<μ,x>
		kappa = self.base_dist.marginal_t.kappa
		mu = self.mu
		return kappa * (mu * x).sum(dim=-1)

	def _log_normalization(self):  # logCp(kappa)
		m = self.mu.shape[-1]
		kappa = self.base_dist.marginal_t.kappa
		return (
				(m / 2 - 1) * torch.log(kappa)
				- (m / 2) * math.log(2 * math.pi)
				- torch.log(ive(m / 2 - 1, kappa)) - kappa
		)


if __name__ == '__main__':
	vmf = VonMisesFisher(func.normalize(torch.randn(3, 8), dim=-1), torch.tensor([1, 2, 3.0]))
	x = vmf.sample(torch.Size([]))
	print(x)
	print(vmf.log_prob(x))
	print(vmf.log_prob_true(x))

	from scipy import stats
	svmf = stats.vonmises_fisher(vmf.mu.numpy(), vmf.kappa.numpy().item())
	print(svmf.logpdf(x.detach().numpy()))
