import numpy as np
import itertools
from typing import Generator, List
import json
from scipy.signal import medfilt2d
import copy
import math
from pathlib import Path
import cv2
import os
from multiprocessing import Pool, cpu_count, set_start_method
from timeit import default_timer as timer
from template import Template

median_filter_rows = 2
reference_generation_hd = 0.4625
noise_ic = Template.from_image(Path("noise_ic.bmp"))
generated_directory = Path("generated")
subjects = 1
cpus = 1
initial_rows, initial_columns = 32, 512
target_rows, target_columns = 64, 512
arch_side_probabilities = (0.5, 0.5, 0.0) # l, r, None
barcode_max_shifts = 10
barcode_mu, barcode_sigma = 6.5, 0.25

with open("exp.json", "r") as f:
	exp_overlaps = json.load(f)
	exp_overlaps = {float(k):float(v) for k,v in exp_overlaps.items()}

class IrisCodeGenerator(object):
	def __init__(self, to_produce, probes_per_subject):
		if not isinstance(to_produce, int) or to_produce < 0:
			raise ValueError("Number of generated iris codes must be a positive integer, instead got:", to_produce)
		self._to_produce = to_produce
		self._probes_per_subject = probes_per_subject

	def __iter__(self):
		self._produced = 0
		return self

	def __next__(self) -> np.ndarray:
		'''Produces one iris-code.'''
		while self._produced < self._to_produce:
			temp_ic = Template.create(initial_rows, initial_columns, barcode_mu, barcode_sigma)
			temp_ic.to_image(Path("test.bmp"))
			temp_ic.shift(np.random.randint(barcode_max_shifts, initial_columns // 2))
			temp_ic.initial_zigzag()

			seqs = temp_ic.find_sequences_of_all(1)
			target_hd = 0.20#float(weibull(2))
			exp_overlap = expected_overlap(target_hd)
			i_hd = (target_hd + exp_overlap) / 2
			b_hd = reference_generation_hd - i_hd

			print ("T-HD:", target_hd)
			print ("I-HD:", i_hd)
			print ("B-HD:", b_hd)
			temp_ic = flip_barcode(temp_ic, b_hd)

			reference, probe, a_hd = flip_templates(temp_ic, target_hd)

			noise_hd = target_hd - a_hd
			print ("A-HD:", a_hd)
			print ("BR-HD:", temp_ic.hamming_distance(reference)[0])
			print ("BP-HD:", temp_ic.hamming_distance(probe)[0])
			print ("N-HD:", noise_hd)
			arch_side = np.random.choice(("l", "r", None), p=arch_side_probabilities)
			print ("AS:", arch_side)
			for ic in (temp_ic, reference, probe):
				ic.add_noise(noise_ic, arch_side, noise_hd if noise_hd > 0 else None)
				ic.remove_top_and_bottom_rows(median_filter_rows)
				ic.expand(2)

			shift = int(np.rint(2 * np.random.randn() + 2))
			print ("S:", shift)
			probe.shift(shift)

			print ("NM-HD:", reference.hamming_distance(probe, rotations=8))
			print ("M-HD:", reference.hamming_distance(probe, rotations=8, mask=True))
			self._produced += 1
			return reference, probe
		else:
			raise StopIteration

def weibull(shape, m=0.0001, t_min=0.015, t_max=0.25):
	normalise = lambda o_min, o_max, t_min, t_max, value: ((t_max - t_min) / (o_max - o_min)) * (value - o_max) + t_max
	X = lambda shape, U: 1.0 * (-np.log2(U)) ** (1 / shape)
	v = X(shape, np.random.rand())
	o_min = X(shape, 1.0)
	o_max = X(shape, m)
	return normalise(o_min, o_max, t_max, t_min, v)

def expected_overlap(target_hd):
	target_hd_r = round(target_hd, 3)
	try:
		exp_overlap = exp_overlaps[target_hd_r]
	except KeyError:
		if target_hd_r < min(exp_overlaps.keys()):
			exp_overlap = min(exp_overlaps.values())
		elif target_hd_r > max(exp_overlaps.keys()):
			exp_overlap = max(exp_overlaps.values())
	return exp_overlap

def flip_barcode(temp_ic, barcode_hd):
	gspace = np.geomspace(0.15, 0.05, num=temp_ic._template.shape[0])
	hd = 0.0
	barcode = copy.deepcopy(temp_ic)
	while not (math.isclose(hd, barcode_hd, abs_tol=0.025) or barcode_hd < hd):
		temp_ic.majority_vote()
		for i, row in enumerate(temp_ic._template):
			temp_ic.flip_edge(row, gspace[i])

		test_ic = copy.deepcopy(temp_ic)
		flip_indices = np.random.randint(low=0, high=temp_ic._template.size-1, size=temp_ic._template.size // 10)
		test_ic._template.flat[flip_indices] ^= 1
		test_ic.medfilt2d()
		hd, _, _ = barcode.hamming_distance(test_ic, 0, cut_rows=median_filter_rows)
	return test_ic

def flip_templates(temp_ic, template_hd):
	gspace = np.geomspace(0.15, 0.05, num=temp_ic._template.shape[0])
	hd = 0.0
	reference, probe = copy.deepcopy(temp_ic), copy.deepcopy(temp_ic)
	while not (math.isclose(hd, template_hd, abs_tol=0.005) or template_hd < hd):
		for template in (reference, probe):
			template.majority_vote()
			template.majority_vote()
			for i, row in enumerate(template._template):
				template.flip_edge(row, gspace[i])
			flip_indices = np.random.randint(low=0, high=template._template.size, size=template._template.size // 20)
			template._template.flat[flip_indices] ^= 1
		test_reference = copy.deepcopy(reference)
		test_probe = copy.deepcopy(probe)
		test_reference.medfilt2d()
		test_probe.medfilt2d()
		hd, _, _ = test_reference.hamming_distance(test_probe, 0, cut_rows=median_filter_rows)
	return test_reference, test_probe, hd
	
def produce(subdirectories: List[int]):
	'''Produce iris-codes segragated into subdirectories.'''
	for subject_num, (reference, probe) in enumerate(IrisCodeGenerator(len(subdirectories), 0)):
		save_dir = generated_directory / Path(str(subdirectories[subject_num]))
		reference.to_image(save_dir / Path("1.bmp"))
		probe.to_image(save_dir / Path("2.bmp"))

if __name__ == '__main__':
	set_start_method("spawn")
	start = timer()
	try:
		num_pools = cpus if cpus <= cpu_count() else cpu_count
	except NotImplementedError:
		num_pools = 1

	subdirectories = np.array_split(range(1, subjects+1), num_pools)
	if num_pools > 1:
		with Pool(num_pools) as p:
			p.map(produce, subdirectories)
	else:
		produce(subdirectories[0])

	stop = timer()
	elapsed = stop - start
	m, s = divmod(elapsed, 60)
	h, m = divmod(m, 60)
	print ("Time elapsed: %d:%02d:%02d" % (h, m, s))