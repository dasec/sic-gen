#!/usr/bin/env python
import argparse
import copy
import cv2
from functools import partial
import itertools
import json
import logging
import math
from multiprocessing import Pool, cpu_count, set_start_method
import numpy as np
from pathlib import Path
from template import Template
from timeit import default_timer as timer
from typing import Generator, Tuple, List, Callable
import validation
from distributions import weibull

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Synthetic iris-code generation.')
parser.add_argument('-n', '--subjects', action='store', type=int, nargs='?', default=1, help='number of subjects (0 < n)')
parser.add_argument('-d', '--directory', action='store', type=Path, nargs='?', default="generated", help='relative path to directory where the generated iris-codes will be stored')
parser.add_argument('-p', '--processes', action='store', type=int, nargs='?', default=1, help='number of CPU processes to use (0 < p, defaults to cpu_count if p > cpu_count or to 1 on unsupported platforms)')
parser.add_argument('-v', '--validate', action='store_true', help='run statistical validation after generation of templates')
parser.add_argument('--version', action='version', version='%(prog)s 0.5')

args = parser.parse_args()
median_filter_rows = 2
majority_vote_rows = 3
split_threshold = 11
reference_generation_hd = 0.4275
noise_ic = Template.from_image(Path("noise_ic.bmp"))
initial_rows, initial_columns = 32, 512
target_rows, target_columns = 64, 512
arch_side_probabilities = (0.45, 0.45, 0.1) # l, r, None
barcode_max_shifts = 10
probe_shifts_mu_sigma = (2,5)
barcode_mu, barcode_sigma = 6.5, 0.25
arch_limits = {"cutin_mul": 0.05, "cutin_add": 0.25, "h_minmax": (10, 25), "w_minmax": (30, 75), "t_minmax": (3,6), "y_minmax": (40, 70)}
flip_gspace = np.geomspace(0.2, 0.15, num=initial_rows + 2 * median_filter_rows)

with open("exp.json", "r") as f:
	exp_overlaps = json.load(f)
	exp_overlaps = {float(k):float(v) for k,v in exp_overlaps.items()}

class IrisCodeGenerator(object):
	def __init__(self, to_produce: int, hd_distribution: Callable[..., float]):
		if not isinstance(to_produce, int) or to_produce < 0:
			raise ValueError("Number of generated iris codes must be a positive integer, instead got:", to_produce)
		self._to_produce = to_produce
		self._hd_distribution = hd_distribution

	def __iter__(self):
		self._produced = 0
		return self

	def __next__(self) -> Tuple[Template, Template]:
		'''Produces synthetic iris-code templates.'''
		while self._produced < self._to_produce:
			# Create, randomly shift and zigzag the initial barcode pattern
			temp_ic = Template.create(initial_rows, initial_columns, barcode_mu, barcode_sigma, median_filter_rows)
			temp_ic.shift(np.random.randint(barcode_max_shifts, initial_columns // 2))
			temp_ic.initial_zigzag()

			# Draw a target HD from the chosen distribution and compute parameters for template generation
			target_hd = self._hd_distribution()
			exp_overlap = expected_overlap(target_hd)
			i_hd = (target_hd + exp_overlap) / 2
			b_hd = reference_generation_hd - i_hd
			logging.debug("Target HD: %f, Barcode HD: %f" % (target_hd, b_hd))

			# Flip barcode bits until an intermediate HD is reached
			temp_ic = flip_barcode(temp_ic, b_hd)

			# Clone the barcode and flip bits until target HD is reached between the reference and probe
			reference, probe, a_hd = flip_templates(temp_ic, target_hd)

			# Generate noise masks and add noise to the templates
			noise_hd = target_hd - a_hd
			arch_side = np.random.choice(("l", "r", None), p=arch_side_probabilities)
			for ic in (reference, probe):
				ic.noise(noise_ic, arch_side, noise_hd if noise_hd > 0 else None, arch_limits if arch_side else None)
				ic.remove_top_and_bottom_rows(median_filter_rows)
				ic.expand(2)

			shift = int(np.rint(probe_shifts_mu_sigma[1] * np.random.randn() + probe_shifts_mu_sigma[0]))
			probe.shift(shift)
			logging.debug("Misalignment: %d" % (shift))

			logging.debug("HD without masks: %f", reference.hamming_distance(probe, rotations=abs(shift))[0])
			logging.debug("HD with masks: %f", reference.hamming_distance(probe, rotations=abs(shift), masks=True)[0])
			self._produced += 1
			return reference, probe
		else:
			raise StopIteration

def expected_overlap(target_hd: float) -> float:
	'''Looks up expected overlap of bit-flips between two templates.'''
	target_hd_r = round(target_hd, 3)
	try:
		exp_overlap = exp_overlaps[target_hd_r]
	except KeyError:
		if target_hd_r < min(exp_overlaps.keys()):
			exp_overlap = min(exp_overlaps.values())
		elif target_hd_r > max(exp_overlaps.keys()):
			exp_overlap = max(exp_overlaps.values())
	return exp_overlap

def flip_barcode(temp_ic: Template, barcode_hd: float) -> Template:
	'''Flips bits in the initial barcode template until an intermediate HD is reached.'''
	hd = 0.0
	barcode = copy.deepcopy(temp_ic)
	while not (math.isclose(hd, barcode_hd, abs_tol=0.01) or barcode_hd < hd):
		temp_ic.majority_vote(majority_vote_rows, split_threshold)
		for i, row in enumerate(temp_ic._template):
			temp_ic.flip_edge(row, flip_gspace[i])
		test_ic = copy.deepcopy(temp_ic)
		flip_indices = np.random.randint(low=0, high=temp_ic._template.size-1, size=temp_ic._template.size // 10)
		test_ic._template.flat[flip_indices] ^= 1
		test_ic.medfilt2d()
		hd, _, _ = barcode.hamming_distance(test_ic, 0, cut_rows=median_filter_rows)
	test_ic.medfilt2d()
	return test_ic

def flip_templates(temp_ic: Template, template_hd: float) -> Tuple[Template, Template, float]:
	'''Takes an intermediate template and produces a reference and probe with a desired HD.'''
	hd = 0.0
	reference, probe = copy.deepcopy(temp_ic), copy.deepcopy(temp_ic)
	while not (math.isclose(hd, template_hd, abs_tol=0.005) or template_hd < hd):
		for template in (reference, probe):
			template.majority_vote(majority_vote_rows, split_threshold)
			template.majority_vote(majority_vote_rows, split_threshold)
			for i, row in enumerate(template._template):
				template.flip_edge(row, flip_gspace[i])
			flip_indices = np.random.randint(low=0, high=template._template.size, size=template._template.size // 100)
			template._template.flat[flip_indices] ^= 1
		test_reference = copy.deepcopy(reference)
		test_probe = copy.deepcopy(probe)
		test_reference.medfilt2d()
		test_probe.medfilt2d()
		hd, _, _ = test_reference.hamming_distance(test_probe, 0, cut_rows=median_filter_rows)
	return test_reference, test_probe, hd
	
def produce(subdirectories: List[int]) -> None:
	'''Produce iris-codes segragated into subdirectories.'''
	for subject_num, (reference, probe) in enumerate(IrisCodeGenerator(len(subdirectories), partial(weibull, 10))):
		save_dir = args.directory / Path(str(subdirectories[subject_num]))
		reference.to_image(save_dir / Path("1.bmp"))
		probe.to_image(save_dir / Path("2.bmp"))
		reference.to_file(save_dir / Path("1.txt"))
		probe.to_file(save_dir / Path("2.txt"))

def validate(processes: int) -> None:
	'''Produces various statistics, which allow to determine whether or not the generated iris-codes have the desired statistical properties.'''
	osiris_interval = [(p.stem[:3], p.stem[-1], Template.from_image(p, None)) for p in sorted(Path("iris_codes_interval").iterdir()) if p.stem[-1] == "1"]
	osiris_biosecure = [(p, p, Template.from_image(p, None)) for p in sorted(Path("iris_codes_biosecure").iterdir()) if p.stem[-1] == "1"]
	files = list(args.directory.glob('**/*.txt'))
	num_files = len(files)
	num_cross_comparisons = num_files * (num_files - 1) // 2
	#if num_cross_comparisons > config.validation_max_comparisons:
	#	num_files = int(math.sqrt(config.validation_max_comparisons) * 2 + 1)
	ic_sample = sorted({(path.parent, path.parent.stem, path.stem.split("_")[0]) for path in itertools.islice(files, num_files)})
	ic_sample = [(p[1], p[2], Template.from_file(p[0] / Path(p[2]+"_template.txt"), p[0] / Path(p[2]+"_mask.txt"))) for p in ic_sample]
	logging.info("Bit counts validation: \nCount: %d \nMin: %.4f \nMax: %.4f \nMean: %.4f \nSt.Dev.: %.4f \nSkew: %.4f \nEx. Kurt.: %.4f" % validation.bit_counts_validation(ic_sample))
	logging.info("Sample of %d from the produced iris-codes selected for validation" % num_files)
	#validation.sequence_lengths_validation(osiris_interval, osiris_biosecure, ic_sample)
	validation.hamming_distance_validation(ic_sample)
	logging.info("Iris-code validation complete")

if __name__ == '__main__':
	start = timer()
	set_start_method("spawn")
	try:
		num_pools = args.processes if args.processes <= cpu_count() else cpu_count
	except NotImplementedError:
		num_pools = 1
	logging.info("Generating iris codes using %u processes" % num_pools)
	logging.info("Storage directory: %s" % args.directory)
	logging.info("Number of subjects: %u" % args.subjects)
	#logging.info("Iris-code size: %u rows, %u columns" % (config.iris_code_rows, config.iris_code_columns))
	subdirectories = np.array_split(range(1, args.subjects+1), num_pools)
	if num_pools > 1:
		with Pool(num_pools) as p:
			p.map(produce, subdirectories)
	else:
		produce(subdirectories[0])
	
	stop = timer()
	elapsed = stop - start
	m, s = divmod(elapsed, 60)
	h, m = divmod(m, 60)
	logging.info("Time elapsed: %d:%02d:%02d" % (h, m, s))
	if args.validate:
		validate(num_pools)