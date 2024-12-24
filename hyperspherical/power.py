import math
from typing import Union

# import power_spherical
import torch
from torch import distributions
from torch.distributions import constraints, kl
from torch.nn import functional as func

from hyperspherical import uniform

_EPS = 1e-36


class _MarginalTDistribution(distributions.TransformedDistribution):
	arg_constraints = {'kappa': constraints.positive}
	has_rsample = True

	def __init__(self, dim, kappa, validate_args=None):
		self.dim = dim
		self.kappa = kappa
		super().__init__(
			distributions.Beta(
				(dim - 1) / 2 + kappa, (dim - 1) / 2, validate_args=validate_args
			),
			transforms=distributions.AffineTransform(loc=-1, scale=2)
		)

	def entropy(self):
		return self.base_dist.entropy() + math.log(2)

	@property
	def mean(self):
		return 2 * self.base_dist.mean - 1

	@property
	def stddev(self):
		return self.variance.sqrt()

	@property
	def variance(self):
		return 4 * self.base_dist.variance


class PowerSpherical(distributions.Distribution):
	arg_constraints = {
		'mu': constraints.real_vector,
		'kappa': constraints.positive
	}
	has_rsample = True
	_epsilon = 1e-36

	def __init__(self, mu: torch.Tensor, kappa, validate_args=None):
		# 预处理 shape
		assert mu.dim() >= 1 and mu.shape[-1] >= 3, '方向维度至少为 3, 二维情况直接用 distributions.VonMises'
		func.normalize(mu.data, dim=-1, out=mu.data)
		if isinstance(kappa, Union[int, float]):
			kappa = torch.full(size=torch.Size(mu.shape[:-1] + (1,)), fill_value=kappa, dtype=mu.dtype, device=mu.device)
		if kappa.dim() == 0 or kappa.shape[-1] != 1:
			kappa.data = kappa.data.unsqueeze(-1)

		# broadcast shapes of mu and kappa
		kappa_shape = kappa.shape[:-1]
		mu_shape = mu.shape[:-1]
		dim_diff = kappa.dim() - mu.dim()
		if dim_diff != 0:
			if dim_diff < 0:
				kappa_shape = torch.Size([1] * dim_diff) + kappa_shape
			else:
				mu_shape = torch.Size([1] * dim_diff) + mu_shape
		broadcast_shape = torch.broadcast_shapes(kappa_shape, mu_shape)
		if kappa_shape != broadcast_shape:
			kappa.data = kappa.data.expand(broadcast_shape + kappa.shape[-1:]).contiguous()
			print(f'broadcast the shape of kappa to {kappa.shape[:-1]}')
		if mu_shape != broadcast_shape:
			mu.data = mu.data.expand(broadcast_shape + mu.shape[-1:]).contiguous()
			print(f'broadcast the shape of mu to {mu.shape}')
		kappa.data = kappa.data.squeeze(-1)
		kappa.data = kappa.data.to(mu.dtype)

		self.mu, self.kappa, = mu, kappa
		super().__init__(batch_shape=mu.shape[:-1], event_shape=mu.shape[-1:], validate_args=validate_args)

		self._dtype = mu.dtype
		self._device = mu.device
		self._m = mu.shape[-1]  # 维度

		# >>> for sampling algorithm >>>
		self._e1 = torch.tensor(  # [1, 0, ..., 0]
			[1.0] + [0] * (self._m - 1), dtype=self._dtype, device=self._device
		)
		self._uniform_spherical = uniform.HypersphericalUniform(
			self._m - 1, batch_shape=self.batch_shape, dtype=self._dtype, device=self._device, validate_args=False
		)
		self._tdist = _MarginalTDistribution(self._m, self.kappa, validate_args=validate_args)

	def sample(self, sample_shape: Union[torch.Size, int] = torch.Size()) -> torch.Tensor:
		if isinstance(sample_shape, int):
			sample_shape = torch.Size([sample_shape])
		with torch.no_grad():  # rsample 是 reparameterized sample, 便于梯度更新以调整分布参数
			return self.rsample(sample_shape)

	def rsample(self, sample_shape=torch.Size()):
		"""
		Reparameterized Sample: 从一个简单的分布通过一个参数化变换使得其满足一个更复杂的分布;
		此处, mu 是可变参数, 通过 radial-tangential decomposition 采样;
		梯度下降更新 mu, 以获得满足要求的 vMF.
		:param sample_shape: 样本的形状
		:return: [shape|m] 的张量, shape 个 m 维方向向量
		"""
		t = self._tdist.rsample(sample_shape).unsqueeze(-1)
		h = torch.clamp(1 - t ** 2, self._epsilon).sqrt()
		v = self._uniform_spherical.sample(sample_shape)
		samples = torch.cat([t, h * v], -1)
		samples = self._householder_rotation(samples)
		return samples

	def _householder_rotation(self, x):
		# 关于 self.mu, 也许只在 rotation 的时候用了一下, 前面的采样估计是按
		# 某个特定的 μ 进行采样, 采好之后, rotate 一下就相当于按 mu 采样了
		# 所以说, 前面那一大坨的计算, 并不涉及 mu 的优化, 它们只是旋转前的 sample, 旋转才是对 mu 梯度有影响的
		# 其实不是什么旋转, 而是一种镜像反射, 应该叫 Householder Transformation
		u = func.normalize(self._e1 - self.mu, dim=-1)  # 要梯度的不要用 .data
		z = x - 2 * (x * u).sum(-1, keepdim=True) * u  # https://zh.wikipedia.org/wiki/豪斯霍尔德变换
		return z

	def log_prob(self, value):
		return self.log_normalizer() + self.kappa * torch.log1p(
			(self.mu * value).sum(-1)
		)

	def log_normalizer(self):
		alpha = (self._m - 1) / 2 + self.kappa
		beta = (self._m - 1) / 2
		return -(
				(alpha + beta) * math.log(2)
				+ torch.lgamma(alpha)
				- torch.lgamma(alpha + beta)
				+ beta * math.log(math.pi)
		)

	def entropy(self):
		alpha = (self._m - 1) / 2 + self.kappa
		beta = (self._m - 1) / 2
		return -(
				self.log_normalizer()
				+ self.kappa
				* (math.log(2) + torch.digamma(alpha) - torch.digamma(alpha + beta))
		)

	@property
	def mean(self):
		return self.mu * self._tdist.mean

	@property
	def stddev(self):
		return self.variance.sqrt()

	@property
	def variance(self):
		alpha = (self._m - 1) / 2 + self.kappa
		beta = (self._m - 1) / 2
		ratio = (alpha + beta) / (2 * beta)
		return self._tdist.variance * (
				(1 - ratio) * self.mu.unsqueeze(-1) @ self.mu.unsqueeze(-2)
				+ ratio * torch.eye(self.mu.shape[-1])
		)


@kl.register_kl(PowerSpherical, uniform.HypersphericalUniform)
def _kl_powerspherical_uniform(p, q):
	return -p.entropy() + q.entropy()


# if __name__ == '__main__':
# 	mu, kappa = torch.randn(10), torch.tensor(3)
# 	pw = power_spherical.PowerSpherical(mu, kappa)
# 	pwn = PowerSpherical(mu, kappa)
#
# 	x = pw.sample(torch.Size([10]))
# 	start = time.time()
# 	samples = pw.sample(torch.Size([10000000]))
# 	print(time.time() - start)
# 	mean = torch.mean(samples, dim=0)
#
# 	print(mean)
# 	print(pw.mean)
# 	print(pwn.mean)
#
# 	print(pw.entropy())
#
# 	cos1 = torch.mean(samples @ mu)
#
# 	start = time.time()
# 	samples = pwn.sample(torch.Size([10000000]))
# 	print(time.time() - start)
# 	print(samples.shape)
# 	cos2 = torch.mean(samples @ mu)
# 	print(cos1, cos2)
#
# 	print(pwn.entropy())
