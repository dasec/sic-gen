__author__ = "Pawel Drozdowski"
__copyright__ = "Copyright (C) 2017 Hochschule Darmstadt"
__license__ = "License Agreement provided by Hochschule Darmstadt(https://github.com/dasec/sic-gen/blob/master/hda-license.pdf)"
__version__ = "1.0"

import itertools
import logging
import numpy as np
from functools import partial
from matplotlib import gridspec
from matplotlib import pyplot as plt
from multiprocessing import Pool
from pathlib import Path
from scipy.stats import describe
from typing import Tuple, List

plt.rc('axes', axisbelow=True)

def distribution_statistics(values: List[float]) -> Tuple[int, float, float, float, float, float, float]:
	'''Produces some basic descriptive statistics for a distribution.'''
	desc = describe(values, nan_policy="omit")
	return desc.nobs, desc.minmax[0], desc.minmax[1], desc.mean, np.sqrt(desc.variance), desc.skewness, desc.kurtosis

def degrees_of_freedom(mu: float, sigma: float) -> int:
	'''Calculates the number of degrees of freedom in a distribution based on its mean and standard deviation.'''
	return int(np.rint((mu * (1 - mu)) / (sigma * sigma)))

def bit_counts_validation(ic_sample: List[np.ndarray]) -> None:
	'''Computes fractions of 0's and 1's in a template sample.'''
	hws = [template[2].hamming_weight() / template[2]._template.size for template in ic_sample]
	return distribution_statistics(hws)

def bit_counts(template) -> Tuple[float, float]:
	'''Computes fractions of 0's and 1's in a template.'''
	hw = template.hamming_weight()
	fraction1 = hw / template._template.size
	fraction0 = 1.0 - fraction1
	return (fraction0, fraction1)

def compute_hd(pair, rotations=0, masks=False):
	'''Helper function for parallelising HD calculations.'''
	return pair[0][2].hamming_distance(pair[1][2], rotations=rotations, masks=masks)[0]

def hamming_distance_validation(processes: int, ic_sample: List[np.ndarray], rotations: int, max_impostor_comparisons: int, path: Path = None) -> None:
	'''Produces Hamming distance histograms and statistics about a sample of Iris-Codes.'''
	all_hds = {}

	# Prepare the pairs of templates to compare
	pairs_genuine = [pair for pair in itertools.combinations(ic_sample, 2) if pair[0][0] == pair[1][0] and pair[0][1] != pair[1][1]]
	logging.debug("Created genuine template pairs")
	pairs_impostor = []
	for pair in itertools.combinations(ic_sample, 2):
		if pair[0][0] != pair[1][0] and pair[0][1] != "2" and pair[1][1] != "2":
			pairs_impostor.append(pair)
			i +=1
		if i >= max_impostor_comparisons:
			break
	logging.debug("Created impostor template pairs")

	# Compute the HDs
	with Pool(processes) as p:
		all_hds["genuine"] = p.map(partial(compute_hd, rotations=0, masks=True), pairs_genuine)
		logging.debug("Genuine HDs calculation finished")
		all_hds["genuine_rot"] = p.map(partial(compute_hd, rotations=rotations, masks=True), pairs_genuine)
		logging.debug("Genuine rotated HDs calculation finished")
		all_hds["impostor"] = p.map(partial(compute_hd, rotations=0, masks=True), pairs_impostor)
		logging.debug("Impostor HDs calculation finished")
		all_hds["impostor_rot"] = p.map(partial(compute_hd, rotations=rotations, masks=True), pairs_impostor)
		logging.debug("Impostor rotated HDs calculation finished")

	# Compute the statistics
	count_g, minimum_g, maximum_g, mu_g, sigma_g, skew_g, kurt_g = distribution_statistics(all_hds["genuine_rot"])
	count_i, minimum_i, maximum_i, mu_i, sigma_i, skew_i, kurt_i = distribution_statistics(all_hds["impostor"])
	count_ir, minimum_ir, maximum_ir, mu_ir, sigma_ir, skew_ir, kurt_ir = distribution_statistics(all_hds["impostor_rot"])
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
	weights = np.ones_like(all_hds["impostor"])/float(len(all_hds["impostor"]))
	plt.hist(all_hds["impostor"], weights=weights, bins=50, color="white", edgecolor="red", alpha=0.75, linewidth=2)
	plt.xlim(0.3, 0.7)
	plt.margins(0.025)
	plt.title("Histogram of impostor HDs without alignment", size=18)
	plt.xlabel("HD", size=16)
	plt.ylabel("Probability Density", size=16)
	plt.grid(True, which="both")

	# Genuine and impostor aligned plot
	ax = plt.subplot(gs[1, :])
	weights_g = np.ones_like(all_hds["genuine_rot"])/float(len(all_hds["genuine_rot"]))
	weights_i = np.ones_like(all_hds["impostor_rot"])/float(len(all_hds["impostor_rot"]))
	plt.hist(all_hds["genuine_rot"], weights=weights_g, bins=50, color="white", edgecolor="green", alpha=0.75, linewidth=2)
	plt.hist(all_hds["impostor_rot"], weights=weights_i, bins=50, color="white", edgecolor="red", alpha=0.75, linewidth=2)
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

	def daugman_hmm(items: int = 1000000) -> List[List[int]]:
		'''Produce sequences from Daugman HMM.'''
		sequences = []
		gen = bit_generator()
		seq = [next(gen)]
		i = 0
		while i < items:
			bit = next(gen)
			if seq[-1] != bit:
				i += 1
				sequences.append(seq)
				seq = [bit]
			else:
				seq.append(bit)
		return sequences

	lengths = list(map(len, daugman_hmm()))
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
	plt.xlabel("Consecutive identical bits", size=16)
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
