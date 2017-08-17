#!/usr/bin/env python
import argparse
import copy
import cv2
from distributions import weibull
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

parser = argparse.ArgumentParser(description='Generator of synthetic Iris-Codes.')
parser.add_argument('-n', '--subjects', action='store', type=int, nargs='?', default=1, help='number of subjects (0 < n)')
parser.add_argument('-d', '--directory', action='store', type=Path, nargs='?', default="generated", help='relative path to directory where the generated iris-codes will be stored')
parser.add_argument('-p', '--processes', action='store', type=int, nargs='?', default=1, help='number of CPU processes to use (0 < p, defaults to cpu_count if p > cpu_count or to 1 on unsupported platforms)')
parser.add_argument('-v', '--validate', action='store_true', help='run statistical validation after generation of templates')
parser.add_argument('-l', '--logging', action='count', help='logging verbosity level')
parser.add_argument('--version', action='version', version='%(prog)s 0.6')
args = parser.parse_args()

if args.logging is not None:
	logging_level = logging.DEBUG if args.logging >= 2 else logging.INFO
else:
	logging_level = logging.WARNING
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging_level)

median_filter_rows = 2
majority_vote_rows = 3
split_threshold = 11
reference_generation_hd = 0.4275
noise_template_target_hd = 0.4
noise_template_refresh_chance = 0.1
initial_rows, initial_columns = 32, 512
target_rows, target_columns = 64, 512
downsampling_every_n_row, downsampling_every_n_column = 8, 2 # 0, 0 to keep the target_rows, target_columns size
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
	'''An iterator for generating synthetic Iris-Codes.'''
	def __init__(self, to_produce: int, hd_distribution: Callable[..., float]):
		self._to_produce = to_produce
		self._hd_distribution = hd_distribution
		self._noise_ic = self._refresh_noise_template()

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

			# Generate fresh noise for the area below eyelid arch
			if np.random.rand() < noise_template_refresh_chance:
				self._noise_ic = self._refresh_noise_template()

			# Add noise and bring the template to standard dimensions (target_rows x target_columns)
			for ic in (reference, probe):
				ic.noise(self._noise_ic, arch_side, noise_hd if noise_hd > 0 else None, arch_limits if arch_side else None)
				ic.remove_top_and_bottom_rows(median_filter_rows)
				ic.expand(2)

			# Apply final misalignment to the probe
			shift = int(np.rint(probe_shifts_mu_sigma[1] * np.random.randn() + probe_shifts_mu_sigma[0]))
			probe.shift(shift)
			logging.debug("Misalignment: %d" % (shift))

			logging.debug("HD without masks: %f", reference.hamming_distance(probe, rotations=abs(shift))[0])
			logging.debug("HD with masks: %f", reference.hamming_distance(probe, rotations=abs(shift), masks=True)[0])
			self._produced += 1
			return reference, probe
		else:
			raise StopIteration

	def _refresh_noise_template(self) -> None:
		'''Generates a new noise template.'''
		logging.debug("Refreshing noise template")
		temp_ic = Template.create(initial_rows, initial_columns, barcode_mu, barcode_sigma, median_filter_rows)
		temp_ic.shift(np.random.randint(barcode_max_shifts, initial_columns // 2))
		temp_ic.initial_zigzag()
		temp_ic = flip_barcode(temp_ic, noise_template_target_hd)
		temp_ic.remove_top_and_bottom_rows(median_filter_rows)
		temp_ic.expand(2)
		return temp_ic

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
	for subject_num, (reference, probe) in enumerate(IrisCodeGenerator(len(subdirectories), partial(weibull, 5))):
		save_dir = args.directory / Path(str(subdirectories[subject_num]))
		reference.to_image(save_dir / Path("1.bmp"))
		probe.to_image(save_dir / Path("2.bmp"))
		reference.to_file(save_dir / Path("1.txt"))
		probe.to_file(save_dir / Path("2.txt"))

def validate(processes: int) -> None:
	'''Produces various statistics, which allow to determine whether or not the generated iris-codes have the desired statistical properties.'''
	logging.info("Validating produced Iris-Codes")
	osiris_interval = [(p.stem[:3], p.stem[-1], Template.from_image(p, None)) for p in sorted(Path("iris_codes_interval").iterdir()) if p.stem[-1] == "1"]
	osiris_biosecure = [(p, p, Template.from_image(p, None)) for p in sorted(Path("iris_codes_biosecure").iterdir()) if p.stem[-1] == "1"]
	files = list(args.directory.glob('**/*.txt'))
	num_files = len(files)
	num_cross_comparisons = num_files * (num_files - 1) // 2
	synthetic_ic = sorted({(path.parent, path.parent.stem, path.stem.split("_")[0]) for path in itertools.islice(files, num_files)})
	synthetic_ic = [(p[1], p[2], Template.from_file(p[0] / Path(p[2]+"_template.txt"), p[0] / Path(p[2]+"_mask.txt"))) for p in synthetic_ic[:25]]
	for dataset in (osiris_interval, osiris_biosecure, synthetic_ic):
		for template in dataset:
			template[2].select(downsampling_every_n_row, downsampling_every_n_column)
	bc_c, bc_max, bc_min, bc_mu, bc_std, bc_sk, bc_ku = validation.bit_counts_validation(synthetic_ic)
	logging.info("Bit counts validation: Count: %d Min: %.4f Max: %.4f Mean: %.4f St.Dev.: %.4f Skewness: %.4f Ex. Kurtosis.: %.4f" % (bc_c, bc_max, bc_min, bc_mu, bc_std, bc_sk, bc_ku))
	validation.sequence_lengths_validation(osiris_interval, osiris_biosecure, synthetic_ic, path=Path("validation_lengths.png"))
	validation.hamming_distance_validation(synthetic_ic, path=Path("validation_hd.png"))
	

if __name__ == '__main__':
	start = timer()
	set_start_method("spawn") # OS interoperability

	# Check CPU availablilty for multiprocessing
	try:
		num_pools = args.processes if args.processes <= cpu_count() else cpu_count
	except NotImplementedError:
		num_pools = 1
	logging.info("Generating iris codes using %u processes" % num_pools)
	logging.info("Storage directory: %s" % args.directory)
	logging.info("Number of subjects: %u" % args.subjects)

	# Multiprocessing
	subdirectories = np.array_split(range(1, args.subjects+1), num_pools)
	if num_pools > 1:
		with Pool(num_pools) as p:
			p.map(produce, subdirectories)
	else: # Single process
		produce(subdirectories[0])

	stop = timer()
	elapsed = stop - start
	m, s = divmod(elapsed, 60)
	h, m = divmod(m, 60)
	print("Time elapsed: %d:%02d:%02d" % (h, m, s))
	if args.validate:
		validate(num_pools)