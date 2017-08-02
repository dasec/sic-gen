import numpy as np
import itertools
from typing import Generator, List
import json
from scipy.signal import medfilt
import copy
import math
from pathlib import Path
import cv2
import os
from typing import Union, Tuple
from multiprocessing import Pool, cpu_count, set_start_method
from timeit import default_timer as timer
from template import Template

median_filter_rows = 2
noise_ic = Template.from_image(Path("noise_ic.bmp"))

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
		return []

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
		for i in range(temp_ic._template.shape[0]):
			previous_b = int(temp_ic._template[i][0])
			for j, b in enumerate(temp_ic._template[i][1:-1], start=1):
				temp_ic.flip_edge(i, j, previous_b, b, gspace[i])
				previous_b = b

		test_ic = copy.deepcopy(temp_ic)
		flip_indices = np.random.randint(low=0, high=temp_ic._template.size-1, size=temp_ic._template.size // 10)
		test_ic._template.flat[flip_indices] ^= 1
		test_ic._template = medfilt(test_ic._template).astype(np.int_)
		test_ic.medfilt()
		hd, _, _ = barcode.hamming_distance(test_ic, 0, cut_rows=median_filter_rows)
	return test_ic

def flip_templates(temp_ic, template_hd):
	gspace = np.geomspace(0.15, 0.05, num=temp_ic._template.shape[0])
	hd = 0.0
	reference, probe = copy.deepcopy(temp_ic), copy.deepcopy(temp_ic)
	while not (math.isclose(hd, template_hd, abs_tol=0.005) or template_hd < hd):
		reference.majority_vote()
		probe.majority_vote()
		for i in range(reference._template.shape[0]):
			previous_b = int(reference._template[i][0])
			for j, b in enumerate(reference._template[i][1:-1], start=1):
				reference.flip_edge(i, j, previous_b, b, gspace[i])
				previous_b = b
		for i in range(probe._template.shape[0]):
			previous_b = int(probe._template[i][0])
			for j, b in enumerate(probe._template[i][1:-1], start=1):
				probe.flip_edge(i, j, previous_b, b, gspace[i])
				previous_b = b
		test_reference = copy.deepcopy(reference)
		test_probe = copy.deepcopy(probe)
		test_reference.medfilt()
		test_probe.medfilt()
		hd, _, _ = test_reference.hamming_distance(test_probe, 0, cut_rows=median_filter_rows)
	return test_reference, test_probe, hd

if __name__ == '__main__':
	set_start_method("spawn")
	reference_generation_hd = 0.4625

	temp_ic = Template.create(32, 512, 6.5, 0.25)
	temp_ic.to_image(Path("test.bmp"))
	temp_ic.shift(np.random.randint(10, 512 // 2))
	temp_ic.initial_zigzag()

	seqs = temp_ic.find_sequences_of_all(1)
	target_hd = float(weibull(6))
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
	arch_side = np.random.choice(("l", "r", None), p=(0.5, 0.5, 0.0))
	print ("AS:", arch_side)
	for ic in (temp_ic, reference, probe):
		ic.add_noise(noise_ic, arch_side, noise_hd if noise_hd > 0 else None)
		ic.remove_top_and_bottom_rows(median_filter_rows)
		ic.expand(2)

	shift = int(np.rint(2 * np.random.randn() + 2))
	print ("S:", shift)
	probe.shift(shift)

	print (reference.hamming_distance(probe, rotations=8))
	print (reference.hamming_distance(probe, rotations=8, mask=True))
	print (probe._template.shape)
	temp_ic.to_image(Path("b.bmp"))
	reference.to_image(Path("r.bmp"))
	probe.to_image(Path("p.bmp"))
	quit()