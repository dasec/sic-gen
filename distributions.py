import numpy as np

def weibull(shape: float, m: float = 0.0001, t_min: float = 0.05, t_max: float = 0.35) -> float:
	'''Returns random values from a normalised Weibull distribution.'''
	normalise = lambda o_min, o_max, t_min, t_max, value: float(((t_max - t_min) / (o_max - o_min)) * (value - o_max) + t_max)
	X = lambda shape, U: 1.0 * (-np.log2(U)) ** (1 / shape)
	v = X(shape, np.random.rand())
	o_min = X(shape, 1.0)
	o_max = X(shape, m)
	val = normalise(o_min, o_max, t_max, t_min, v)
	while not (t_min < val < t_max):
		val = normalise(o_min, o_max, t_max, t_min, v)
	return val