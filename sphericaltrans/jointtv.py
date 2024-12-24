import torch
from torch import distributions

from sphericaltrans.uniform import HypersphericalUniform


class JointTSDistribution(distributions.Distribution):
	def __init__(self, marginal_t, marginal_s: HypersphericalUniform):
		assert marginal_t.batch_shape == marginal_s.batch_shape
		super().__init__(batch_shape=marginal_s.batch_shape, event_shape=torch.Size([marginal_t.dim]), validate_args=False)
		self.marginal_t, self.marginal_s = marginal_t, marginal_s

	def rsample(self, sample_shape=torch.Size()):
		return torch.cat(
			[
				self.marginal_t.rsample(sample_shape).unsqueeze(-1),
				self.marginal_s.sample(sample_shape),
			],
			-1
		)

	def log_prob(self, value):
		return self.marginal_t.log_prob(value[..., 0]) + self.marginal_s.log_prob(value[..., 1:])

	def entropy(self):
		return self.marginal_t.entropy() + self.marginal_s.entropy()
