from matplotlib import gridspec
from matplotlib import pyplot as plt
import logging
import numpy as np
from pathlib import Path
from scipy.stats import describe
from typing import Tuple, List

plt.rc('axes', axisbelow=True)

def distribution_statistics(values: List[float]) -> Tuple[int, float, float, float, float, float, float]:
	'''Produces some basic descriptive statistics for a distribution.'''
	desc = describe(values)
	return desc.nobs, desc.minmax[0], desc.minmax[1], desc.mean, np.sqrt(desc.variance), desc.skewness, desc.kurtosis

def degrees_of_freedom(mu: float, sigma: float) -> int:
	'''Calculates the number of degrees of freedom in a distribution based on its mean and standard deviation.'''
	return int(np.rint((mu * (1 - mu)) / (sigma * sigma)))

def bit_counts_validation(ic_sample: List[np.ndarray]) -> None:
	hws = [template[2].hamming_weight() / template[2]._template.size for template in ic_sample]
	return distribution_statistics(hws)

def bit_counts(template) -> Tuple[float, float]:
	'''Computes fractions of 0's and 0's in a template.'''
	hw = template.hamming_weight()
	fraction1 = hw / template._template.size
	fraction0 = 1.0 - fraction1
	return (fraction0, fraction1)

def hamming_distance_validation(ic_sample: List[np.ndarray], path: Path = None) -> None:
	'''Produces Hamming distance histograms and statistics about a sample of Iris-Codes.'''
	hds_genuine = []
	hds_impostor = []
	hds_genuine_rot = []
	hds_impostor_rot = []
	done = set()
	rotations = 10

	# Compute HDs
	for i in range(len(ic_sample)):
		s1, i1, ic1 = ic_sample[i]
		for j in range(i+1, len(ic_sample)):
			s2, i2, ic2 = ic_sample[j]
			hd, _, _ = ic1.hamming_distance(ic2, rotations=0, masks=True)
			hd_rot, _, _ = ic1.hamming_distance(ic2, rotations=rotations, masks=True)
			if s1 != s2 and ((s1, i1), (s2, i2)) not in done and ((s2, i2), (s1, i1)) not in done and i1 != "2" and i2 != "2":
				hds_impostor.append(hd)
				hds_impostor_rot.append(hd_rot)
			elif s1 == s2:
				hds_genuine.append(hd)
				hds_genuine_rot.append(hd_rot)
			done.add(((s1, i1), (s2, i2)))
			done.add(((s2, i2), (s1, i1)))

	# Compute the statistics
	count_g, minimum_g, maximum_g, mu_g, sigma_g, skew_g, kurt_g = distribution_statistics(hds_genuine_rot)
	count_i, minimum_i, maximum_i, mu_i, sigma_i, skew_i, kurt_i = distribution_statistics(hds_impostor)
	count_ir, minimum_ir, maximum_ir, mu_ir, sigma_ir, skew_ir, kurt_ir = distribution_statistics(hds_impostor_rot)
	df_i = degrees_of_freedom(mu_i, sigma_i)
	logging.info("Impostor unaligned distribution: Count: %d Min: %.5f Max: %.5f Mean: %.5f St.Dev.: %.5f Skewness: %.5f Ex. Kurtosis: %.5f, Degrees of freedom: %d" % (count_i, minimum_i, maximum_i, mu_i, sigma_i, skew_i, kurt_i, df_i))
	logging.info("Impostor aligned distribution: Count: %d Min: %.5f Max: %.5f Mean: %.5f St.Dev.: %.5f Skewness: %.5f Ex. Kurtosis: %.5f" % (count_ir, minimum_ir, maximum_ir, mu_ir, sigma_ir, skew_ir, kurt_ir))
	logging.info("Genuine aligned distribution: Count: %d Min: %.5f Max: %.5f Mean: %.5f St.Dev.: %.5f Skewness: %.5f Ex. Kurtosis: %.5f" % (count_g, minimum_g, maximum_g, mu_g, sigma_g, skew_g, kurt_g))

	# Create the plotting grid
	fig = plt.figure(figsize=(15,10))
	gs = gridspec.GridSpec(2,2)
	fig.subplots_adjust(hspace=0.25)

	# Impostor unaligned plot
	ax = plt.subplot(gs[0, :])
	weights = np.ones_like(hds_impostor)/float(len(hds_impostor))
	plt.hist(hds_impostor, weights=weights, bins=50, color="white", edgecolor="red", alpha=0.75, linewidth=2)
	plt.xlim(0.3, 0.7)
	plt.margins(0.025)
	plt.title("Histogram of impostor HDs without alignment", size=18)
	plt.xlabel("HD", size=16)
	plt.ylabel("Probability Density", size=16)
	plt.grid(True, which="both")

	# Genuine and impostor aligned plot
	ax = plt.subplot(gs[1, :])
	weights_g = np.ones_like(hds_genuine_rot)/float(len(hds_genuine_rot))
	weights_i = np.ones_like(hds_impostor_rot)/float(len(hds_impostor_rot))
	plt.hist(hds_genuine_rot, weights=weights_g, bins=50, color="white", edgecolor="green", alpha=0.75, linewidth=2)
	plt.hist(hds_impostor_rot, weights=weights_i, bins=50, color="white", edgecolor="red", alpha=0.75, linewidth=2)
	plt.margins(0.025)
	plt.title("Histograms of all HDs with alignment ($\pm$%d bits)" % rotations, size=18)
	plt.xlabel("HD", size=16)
	plt.ylabel("Probability Density", size=16)
	plt.grid(True, which="both")

	if path is None:
		plt.show()
		plt.clf()
	else:
		plt.savefig(str(path), format=path.suffix.replace(".", ""), bbox_inches='tight')

def sequence_lengths_validation(*ic_samples: List[np.ndarray], path: Path = None) -> None:
	'''Produces a histogram of lengths of consecutive bits sequences for the given sample of iris-codes.'''
	def bit_generator():
		'''Yields Iris-Code bits according to Daugman's HMM.'''
		a = 90
		previous = np.random.choice([0,1])
		while True:
			choices = [previous] * a + [previous^1] * (100 - a)
			choice = np.random.choice(choices)
			yield choice
			previous = choice

	# Produce sequences from Daugman HMM
	sequences = []
	gen = bit_generator()
	seq = [next(gen)]
	i = 0
	while i < 100000:
		bit = next(gen)
		if seq[-1] != bit:
			i += 1
			sequences.append(seq)
			seq = [bit]
		else:
			seq.append(bit)
	lengths = list(map(len, sequences))
	count, minimum, maximum, mu, sigma, skew, kurt = distribution_statistics(lengths)
	logging.info("Sequence lengths %s: Count: %d Min: %d Max: %d Mean: %.5f St.Dev.: %.5f Skewness: %.5f Ex. Kurtosis: %.5f" % ("Daugman's HMM", count, minimum, maximum, mu, sigma, skew, kurt))
	y, x = np.histogram(lengths, density=True, bins=range(1, 30 + 3))
	plt.plot(x[:-2], y[:-1], marker="o", linewidth=2, color="C3", alpha=0.75, label="Daugman's HMM")

	# Produce sequences from other datasets
	markers = ["^", "v", "x"]
	labels = ["OSIRIS Interval", "OSIRIS Biosecure", "SIC-Gen"]
	for i, ic_sample in enumerate(ic_samples):
		all_sequences_lengths = []
		for ic in ic_sample:
			seq1 = ic[2].find_sequences_of_all(1)
			seq0 = ic[2].find_sequences_of_all(0)
			seqs = np.concatenate((seq0, seq1))
			all_sequences_lengths += [end - start for start, end in seqs]
		count, minimum, maximum, mu, sigma, skew, kurt = distribution_statistics(all_sequences_lengths)
		logging.info("Sequence lengths %s: Count: %d Min: %d Max: %d Mean: %.5f St.Dev.: %.5f Skewness: %.5f Ex. Kurtosis: %.5f" % (labels[i], count, minimum, maximum, mu, sigma, skew, kurt))
		y, x = np.histogram(all_sequences_lengths, density=True, bins=range(1, 30 + 2))
		plt.plot(x[:-1], y, marker=markers[i], linewidth=2, color="C{}".format(i), alpha=0.75, label=labels[i])
	plt.title("Distribution of sequence lengths", size=18)
	plt.xlabel("Consecutive bits", size=16)
	plt.ylabel("Probability density", size=16)
	plt.xticks([1, 5, 10, 15, 20, 25, 30])
	plt.xlim(0, 30.75)
	plt.legend(loc=0)
	plt.grid(True)
	plt.margins(0.025)

	if path is None:
		plt.show()
		plt.clf()
	else:
		plt.savefig(str(path), format=path.suffix.replace(".", ""), bbox_inches='tight')