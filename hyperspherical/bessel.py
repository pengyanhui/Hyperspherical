from numbers import Number

import numpy as np
from scipy import special
import torch
from torch import autograd, nn


class IveFunction(autograd.Function):
	"""
	实现 Bessel Function 的梯度计算, 属于"用了其他库"的情况
	项目中应该是为了优化 kappa;

	注意这里 ive(v, z) = iv(v, z) * exp(-abs(z.real))
	关于 Bessel Function 的各种公式, 参见 https://en.wikipedia.org/wiki
	"""

	@staticmethod
	def forward(ctx, *args):
		v, z = args  # v 是阶数, z 是函数变量
		assert isinstance(v, Number) and isinstance(z, torch.Tensor) and z.is_floating_point(), \
			'v must be a scalar, and z must be a tensor of floating point type'
		z_np = z.data.cpu().numpy()

		if np.isclose(np.array(v), 0):  # 求 0 阶
			output = special.i0e(z_np)  # 计算修正的零阶贝塞尔函数
		elif np.isclose(np.array(v), 1):  # 1 阶
			output = special.i1e(z_np)
		else:  # v > 0  v=p/2, p 是向量维度, 往往很高, 故而阶数还是比较大的
			output = special.ive(v, z_np)
		ctx.v = v  # Non-tensors should be stored directly on ctx
		ctx.z = z_np
		ctx.ive = output
		ctx.dtype = z.dtype
		ctx.device = z.device
		return torch.tensor(output, dtype=z.dtype, device=z.device)  # 算好 Ive(z) 后, 转回 Tensor

	@staticmethod
	def backward(ctx, *grad_output):
		grad_v = None  # 如果不需要梯度, 或者 input 不是 Tensor, 那么请返回 None, 这里对应阶数 v
		# 这里递归调用 ive 会有问题吗? => 时只计算 forward, <= 时计算 backward
		# 如果能正常运行, 说明 backward 中的 ive 只参与计算, 而不管内层的 backward
		# 所以内层的 ive 应该不会对梯度产生影响
		grad_z = special.ive(ctx.v - 1, ctx.z) - ctx.ive * (ctx.v + ctx.z) / ctx.z  # 关于 kappa 的导数吗?
		# 返回数量要和 forward 的输入保持一致
		return grad_v, grad_output[0] * torch.tensor(grad_z, dtype=ctx.dtype, device=ctx.device)


class Ive(nn.Module):
	def __init__(self, v):
		super(Ive, self).__init__()
		self.v = v

	def forward(self, z):
		return IveFunction.apply(self.v, z)


ive = IveFunction.apply
