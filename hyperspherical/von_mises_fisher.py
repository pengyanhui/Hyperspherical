import math
from typing import Union

import torch
from torch import distributions
from torch.distributions import constraints
from torch.distributions.kl import register_kl
from torch.nn import functional as func

from hyperspherical.bessel import ive  # 采样过程并没有用到 ive, 所以我扯那一拨关于 Bessel Function 的梯度问题并没有用.
from hyperspherical.uniform import HypersphericalUniform


# 其实 torch 里有 ive, 只不过只能 i0e 和 i1e
# from torch import special
# x = special.i0e(torch.tensor(1))
# 且能进行梯度计算


class VonMisesFisher(distributions.Distribution):
	"""
	scipy 的 stats.vonmises_fisher 实现了 vMF 分布, 官方文档说它也是根据这个采样算法实现的;
	经测验, 当使用 CPU 时, 本实现比 scipy 采样略慢; 但本实现在 GPU 上采样相当快, 而 scipy 无法在 GPU 上运算;
	且本实现能够设置参数为可优化 require_grad=True.

	一般来说, 维度 p 固定了, 那么优化的参数就是 kappa 和 mu 了;
	mu 不在 Bessel Function I{p/2-1} 中, 所以梯度计算简单, PyTorch 可自己搞定;
	kappa 是 Ip/2-1(k) 的参数, 不可导, 则计算梯度需要用户编写 autograd.Function;
	"""
	arg_constraints = {  # 对参数的一些限制, 如果 self.xxx 没有被设置为被限制的类型, 则报错
		'mu': constraints.real_vector,
		'kappa': constraints.positive
	}
	support = constraints.real_vector  # 支撑集
	has_rsample = True

	_epsilon = 1e-36

	def __init__(self, mu: torch.Tensor, kappa: Union[torch.Tensor, float], validate_args=None):
		"""
		:param mu: μ 待优化
		:param kappa: kappa
		:param validate_args: 是否检查类型限制
		"""
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
		# kappa.data = kappa.data.to(mu.dtype)

		self.mu = mu
		self.kappa = kappa
		# 这个之所以放到后面, 是因为 Distribution 的 __init__ 会检查参数的合法性, 如果放在前面, 可能会找不到 'mu' 和 'kappa'
		super().__init__(mu.shape[:-1], mu.shape[-1:], validate_args=validate_args)

		self._dtype = mu.dtype
		self._device = mu.device
		self._m = mu.shape[-1]  # 维度

		# >>> for sampling algorithm >>>
		self._e1 = torch.tensor(  # [1, 0, ..., 0]
			[1.0] + [0] * (self._m - 1), dtype=self._dtype, device=self._device
		)
		self._normal = distributions.MultivariateNormal(
			torch.zeros(self._m - 1, dtype=self._dtype, device=self._device),
			torch.eye(self._m - 1, dtype=self._dtype, device=self._device)
		)
		self._uniform = distributions.Uniform(self._epsilon, torch.tensor(1.0 - self._epsilon, dtype=self._dtype, device=self._device))
		self._beta = distributions.Beta(
			torch.tensor((self._m - 1) / 2, dtype=self._dtype, device=self._device),
			torch.tensor((self._m - 1) / 2, dtype=self._dtype, device=self._device)
		)
		# <<< for sampling algorithm <<<

	@property
	def mean(self):
		# mean 不应该是 mu=μ 吗? hhh!!! mean 和 mean direction 不是一回事
		return self._apk().unsqueeze(-1) * self.mu

	def _apk(self):
		"""
		当贝塞尔函数阶数过高(超过50), 同时 kappa 较小(<1), 会出现数值不稳定的情况;
		阶数越高, 需要 kappa 也就随之上升, 比如阶数 >80 时, kappa=2 就会不稳;
		"""
		return ive(self._m / 2, self.kappa) / ive((self._m / 2) - 1, self.kappa)

	def sample(self, shape: Union[torch.Size, int] = torch.Size()):
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
		if isinstance(shape, int):
			shape = torch.Size([shape])
		# shape = torch.Size(shape + self._batch_shape + self._event_shape)
		t = (
			self._sample_t3(shape=shape)
			if self._m == 3
			else self._sample_t_rej(shape=shape)
		).unsqueeze(-1)
		v = self._normal.sample(torch.Size(shape + self._batch_shape))
		func.normalize(v.data, dim=-1, out=v.data)

		h = torch.clamp(1 - t ** 2, self._epsilon).sqrt()
		samples = torch.cat([t, h * v], -1)
		samples = self._householder_rotation(samples)
		return samples

	def _sample_t3(self, shape: torch.Size):
		"""
		拒绝采样方法来自: Computer Generation of Distributions on the M-Sphere
		https://rss.onlinelibrary.wiley.com/doi/abs/10.2307/2347441
		:param shape: 采样 t 的的形状
		:return: 形状为 shape 的张量, shape 个 t
		"""
		# kappa 也是有 shape 的, 说明可以并行多个 κ 吗?
		shape = torch.Size(shape + self.kappa.shape)  # torch.Size 继承自 tuple, 其 + 运算就是连接操作
		# https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution # 3-D sphere
		u = self._uniform.sample(shape)
		t = 1 + torch.stack(  # 这个公式是按 μ=(0,0,1) 计算的 t, arccosw=φ, 即 t=z
			[  # 最后的旋转可能是旋转至按真正的 μ 采样结果
				torch.log(u),
				torch.log(1 - u) - 2 * self.kappa
			],
			dim=0
		).logsumexp(0) / self.kappa
		return t

	def _sample_t_rej(self, shape: torch.Size):
		num_need = math.prod(shape)
		num_kappa = math.prod(self.kappa.shape)
		sample_shape = torch.Size([num_kappa, 10 + math.ceil(num_need * 1.3)])

		kappa = self.kappa.reshape(-1, 1)
		c = torch.sqrt(4 * kappa.square() + (self._m - 1) ** 2)
		b = (-2 * kappa + c) / (self._m - 1)
		t_0 = (-(self._m - 1) + c) / (2 * kappa)
		s = kappa * t_0 + (self._m - 1) * torch.log(1 - t_0.square())

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
			mask = (kappa * t + (self._m - 1) * torch.log(1 - t_0 * t) - s) > torch.log(u)
			for i in range(num_kappa):
				ts[i].append(t[i][mask[i]])
			cnts += mask.sum(dim=-1)
		ts = torch.stack([torch.cat(t)[:num_need] for t in ts], dim=1)
		return ts.reshape(shape + self.kappa.shape)

	def _householder_rotation(self, x):
		# 关于 self.mu, 也许只在 rotation 的时候用了一下, 前面的采样估计是按
		# 某个特定的 μ 进行采样, 采好之后, rotate 一下就相当于按 mu 采样了
		# 所以说, 前面那一大坨的计算, 并不涉及 mu 的优化, 它们只是旋转前的 sample, 旋转才是对 mu 梯度有影响的
		# 其实不是什么旋转, 而是一种镜像反射, 应该叫 Householder Transformation
		u = func.normalize(self._e1 - self.mu, dim=-1)  # 要梯度的不要用 .data
		z = x - 2 * (x * u).sum(-1, keepdim=True) * u  # https://zh.wikipedia.org/wiki/豪斯霍尔德变换
		return z

	def entropy(self):
		return -self.kappa * self._apk() - self._log_normalization()

	def log_prob(self, x):
		if self._validate_args:
			self._validate_sample(x)
		return self._log_unnormalized_prob(x) + self._log_normalization()

	def _log_unnormalized_prob(self, x):  # k<μ,x>
		return self.kappa * torch.sum(self.mu * x, dim=-1)

	def _log_normalization(self):  # logCp(kappa)
		return (
				(self._m / 2 - 1) * self.kappa.log()
				- (self._m / 2) * math.log(2 * math.pi)
				- (ive(self._m / 2 - 1, self.kappa).log() + self.kappa)
		)


@register_kl(VonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(vmf, hyu):
	return -vmf.entropy() + hyu.entropy()  # √


if __name__ == '__main__':
	# 经验证, 基本和 scipy.stats.vonmises_fisher 一致
	mu = torch.randn(6)
	mu = func.normalize(mu, dim=-1)
	kappa = torch.arange(1, 8)
	vmf = VonMisesFisher(mu, 50)
	samples = vmf.sample(torch.Size([1000000]))
	print(samples.mean(dim=[0]))
	print(vmf.mean)
	print(vmf.mu)
	print(vmf.sample())
	print(vmf.entropy())
	print(vmf.log_prob(samples).shape)
