import torch
from torch import distributions
from torch.distributions import constraints
from torch.nn import functional as func

_EPS = 1e-36


class TTransform(distributions.Transform):
	domain = constraints.real_vector
	codomain = constraints.real_vector

	def _call(self, x):
		t = x[..., :1]
		v = x[..., 1:]
		return torch.cat((t, v * torch.sqrt(torch.clamp(1 - t ** 2, _EPS))), -1)

	def _inverse(self, y):
		t = y[..., :1]
		v = y[..., 1:]
		return torch.cat([t, v / torch.sqrt(torch.clamp(1 - t ** 2, _EPS))], -1)

	def log_abs_det_jacobian(self, x, y):
		t = x[..., 0]  # 经试验, 确实是 (m-3)/2 正确, 但我不知道为啥, 放弃了
		return ((x.shape[-1] - 3) / 2) * torch.log(torch.clamp(1 - t ** 2, _EPS))


class HouseholderRotationTransform(distributions.Transform):
	domain = constraints.real_vector
	codomain = constraints.real_vector

	def __init__(self, loc):
		super().__init__()
		func.normalize(loc.data, dim=-1, eps=_EPS, out=loc.data)
		self._loc = loc
		self._e1 = torch.zeros_like(loc)
		self._e1.data[..., 0] = 1.0

	def _call(self, x):
		# 如果想产生梯度, 结算部分都要放到计算图中
		u = func.normalize(self._e1 - self._loc, dim=-1, eps=_EPS)
		return x - 2 * (x * u).sum(-1, keepdim=True) * u

	def _inverse(self, y):
		u = func.normalize(self._e1 - self._loc, dim=-1, eps=_EPS)
		return y - 2 * (y * u).sum(-1, keepdim=True) * u

	def log_abs_det_jacobian(self, x, y):
		return torch.zeros_like(x[..., 0])
