import numpy as np
from typing import Union, Tuple, List
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab

Number = Union[float, int]

def distribution_statistics(values: List[Number]) -> Tuple[Number, Number, Number, Number, Number]:
	'''Produces some basic descriptive statistics for a distribution.'''
	return len(values), min(values), max(values), np.mean(values), np.std(values)

def degrees_of_freedom(mu: Number, sigma: Number) -> int:
	'''Calculates the number of degrees of freedom in a distribution based on its mean and standard deviation.'''
	return int(np.rint((mu * (1 - mu)) / (sigma * sigma)))

def bit_counts(template):
	hw = template.hamming_weight()
	fraction1 = hw / template._template.size
	fraction0 = 1.0 - fraction1
	return (fraction0, fraction1)

def hamming_distance_validation(ic_sample: List[np.ndarray], path: Path = None) -> None:
	'''Produces a histogram and statistics about a collection of HD scores.'''
	imp_comparisons = 0
	gen_comparisons = 0
	hds_genuine = []
	hds_impostor = []
	done = set()
	print (ic_sample)
	for i in range(len(ic_sample)):
		print (i)
		s1, i1, ic1 = ic_sample[i]
		for j in range(i+1, len(ic_sample)):
			s2, i2, ic2 = ic_sample[j]
			hd, _, _ = ic1.hamming_distance(ic2, rotations=0, masks=True)
			if s1 != s2 and ((s1, i1), (s2, i2)) not in done and ((s2, i2), (s1, i1)) not in done and i1 != "2" and i2 != "2":
				hds_impostor.append(hd)
				imp_comparisons += 1
			elif s1 == s2:
				hds_genuine.append(hd)
				gen_comparisons += 1
			done.add(((s1, i1), (s2, i2)))
			done.add(((s2, i2), (s1, i1)))
	count_g, minimum_g, maximum_g, mu_g, sigma_g = distribution_statistics(hds_genuine)
	count_i, minimum_i, maximum_i, mu_i, sigma_i = distribution_statistics(hds_impostor)
	df = degrees_of_freedom(mu_i, sigma_i)
	plt.hist(hds_genuine, normed=True, bins=50, color="white", edgecolor="green", alpha=0.75, linewidth=2)
	plt.hist(hds_impostor, normed=True, bins=25, color="white", edgecolor="red", alpha=0.75, linewidth=2)
	box_text = '$N=%d$\n$\mu=%.5f$\n$\sigma=%.5f$\n$\mathit{min}=%.5f$\n$\mathit{max}=%.5f$\n$\\nu=%d$' % (count_i, mu_i, sigma_i, minimum_i, maximum_i, df)
	plt.text(0.77, 0.975, box_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75))
	box_text = '$\\mathbf{Genuine}$\n$N=%d$\n$\mu=%.5f$\n$\sigma=%.5f$\n$\mathit{min}=%.5f$\n$\mathit{max}=%.5f$' % (count_g, mu_g, sigma_g, minimum_g, maximum_g)
	plt.text(0.02, 0.975, box_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75))
	x = np.linspace(0.0, 1.0, 1000)
	plt.plot(x, mlab.normpdf(x, mu_i, sigma_i), linestyle="--", color="blue", linewidth=2)
	plt.title("Histogram of HDs", size=18)
	plt.xlabel("HD", size=16)
	plt.ylabel("Probability Density", size=16)
	plt.xlim(0.3, 0.7)
	plt.grid(True)
	if path is None:
		plt.show()
		plt.clf()
	else:
		plt.savefig(path)

def sequence_lengths_validation(*ic_samples: List[np.ndarray], path: Path = None) -> None:
	'''Produces a histogram of lengths of consecutive bits sequences for the given sample of iris-codes.'''
	def bit_generator():
		a = 90
		previous = np.random.choice([0,1])
		while True:
			choices = [previous] * a + [previous^1] * (100 - a)
			choice = np.random.choice(choices)
			yield choice
			previous = choice
	sequences = []
	gen = bit_generator()
	seq = [next(gen)]
	i = 0
	while i < 10000:
		#print (i)
		bit = next(gen)
		if seq[-1] != bit:
			i += 1
			#print (seq, len(seq))
			sequences.append(seq)
			seq = [bit]
		else:
			seq.append(bit)
	lengths = list(map(len, sequences))
	y, x = np.histogram(lengths, density=True, bins=range(1, 30 + 3))
	plt.plot(x[:-2], y[:-1], marker="o", linewidth=2, color="C3", alpha=0.75, label="Daugman's HMM")
	markers = ["^", "v", "x"]
	labels = ["OSIRIS Interval", "OSIRIS Biosecure", "SIC-Gen"]
	for i, ic_sample in enumerate(ic_samples):
		all_sequences_lengths = []
		for ic in ic_sample:
			seq1 = ic[2].find_sequences_of_all(1)
			seq0 = ic[2].sequences_of_0_from_sequences_of_1(seq1)
			seqs = np.concatenate((seq0, seq1))
			all_sequences_lengths += [end - start for start, end in seqs]
		count, minimum, maximum, mu, sigma = distribution_statistics(all_sequences_lengths)
		y, x = np.histogram(all_sequences_lengths, density=True, bins=range(1, 30 + 2))
		plt.plot(x[:-1], y, marker=markers[i], linewidth=2, color="C{}".format(i), alpha=0.75, label=labels[i])
			#plt.hist(all_sequences_lengths, normed=True, bins=range(1, 30 + 1), color="white", edgecolor="C{}".format(i), alpha=0.75, linewidth=2)
	#box_text = '$\mu=%.5f$\n$\sigma=%.5f$\n$\mathit{min}=%d$\n$\mathit{max}=%d$' % (mu, sigma, minimum, maximum)
	#plt.text(0.815, 0.975, box_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.75))
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
		plt.savefig(path)