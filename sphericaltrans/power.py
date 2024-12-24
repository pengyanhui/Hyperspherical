import math
from numbers import Number

import torch
from torch import distributions
from torch.distributions import constraints, kl
from torch.nn import functional as func

from sphericaltrans.jointtv import JointTSDistribution
from sphericaltrans.uniform import HypersphericalUniform
from sphericaltrans.transforms import TTransform, HouseholderRotationTransform


class _MarginalTDistribution(distributions.TransformedDistribution):
	arg_constraints = {'kappa': constraints.positive}

	def __init__(self, dim, kappa, validate_args=None):
		self.dim = dim
		self.kappa = kappa  # 仅用于 argument check 了

		# batch_shape = kappa.shape
		# event_shape = []
		super().__init__(
			distributions.Beta((dim - 1) / 2 + kappa, (dim - 1) / 2, validate_args=validate_args),
			transforms=distributions.AffineTransform(loc=-1, scale=2)
		)

	def entropy(self):
		return self.base_dist.entropy() + math.log(2)  # √

	@property
	def mean(self):
		return 2 * self.base_dist.mean - 1

	@property
	def stddev(self):
		return self.variance.sqrt()

	@property
	def variance(self):
		return 4 * self.base_dist.variance


class PowerSpherical(distributions.TransformedDistribution):
	arg_constraints = {
		'mu': constraints.real_vector,
		'kappa': constraints.positive,
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
			kappa = torch.tensor([kappa], dtype=mu.dtype, device=mu.device)

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
			[TTransform(), HouseholderRotationTransform(mu)],
		)

	def log_prob_true(self, value):
		return self.log_normalizer() + self.kappa * (self.mu * value).sum(-1).log1p()

	def log_normalizer(self):
		alpha = self.base_dist.marginal_t.base_dist.concentration1
		beta = self.base_dist.marginal_t.base_dist.concentration0
		return -(
				(alpha + beta) * math.log(2)
				+ torch.lgamma(alpha)
				- torch.lgamma(alpha + beta)
				+ beta * math.log(math.pi)
		)

	def entropy(self):
		alpha = self.base_dist.marginal_t.base_dist.concentration1
		beta = self.base_dist.marginal_t.base_dist.concentration0
		return -(
				self.log_normalizer()
				+ self.kappa
				* (math.log(2) + torch.digamma(alpha) - torch.digamma(alpha + beta))
		)

	@property
	def mean(self):
		return self.mu * self.base_dist.marginal_t.mean.unsqueeze(-1)

	@property
	def stddev(self):
		return self.variance.sqrt()

	@property
	def variance(self):
		alpha = self.base_dist.marginal_t.base_dist.concentration1
		beta = self.base_dist.marginal_t.base_dist.concentration0
		ratio = (alpha + beta) / (2 * beta)
		return self.base_dist.marginal_t.variance * (
				(1 - ratio) * self.mu.unsqueeze(-1) @ self.mu.unsqueeze(-2)
				+ ratio * torch.eye(self.mu.shape[-1])
		)


@kl.register_kl(PowerSpherical, HypersphericalUniform)
def _kl_powerspherical_uniform(p, q):
	return -p.entropy() + q.entropy()


if __name__ == '__main__':
	ps = PowerSpherical(torch.ones(5), torch.tensor(6.0))
	x = ps.sample(torch.Size([]))
	print(x.shape)
	print(x.norm(dim=-1).shape)
	print(x.mean(dim=0))
	print(ps.mean)
	print(ps.log_prob(x))
	print(ps.log_prob_true(x))
