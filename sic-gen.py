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
			temp_ic = Template.create(initial_rows, initial_columns, barcode_mu, barcode_sigma)
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
			arch_side = "l"#np.random.choice(("l", "r", None), p=arch_side_probabilities)
			for ic in (reference, probe):
				ic.noise(noise_ic, arch_side, noise_hd if noise_hd > 0 else None)
				ic.remove_top_and_bottom_rows(median_filter_rows)
				ic.expand(2)

			shift = int(np.rint(2 * np.random.randn() + 2))
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
	gspace = np.geomspace(0.2, 0.15, num=temp_ic._template.shape[0])
	hd = 0.0
	barcode = copy.deepcopy(temp_ic)
	while not (math.isclose(hd, barcode_hd, abs_tol=0.01) or barcode_hd < hd):
		temp_ic.majority_vote()
		for i, row in enumerate(temp_ic._template):
			temp_ic.flip_edge(row, gspace[i])
		#print (validation.bit_counts(temp_ic))
		test_ic = copy.deepcopy(temp_ic)
		flip_indices = np.random.randint(low=0, high=temp_ic._template.size-1, size=temp_ic._template.size // 10)
		test_ic._template.flat[flip_indices] ^= 1
		test_ic.medfilt2d()
		hd, _, _ = barcode.hamming_distance(test_ic, 0, cut_rows=median_filter_rows)
	test_ic.medfilt2d()
	return test_ic

def flip_templates(temp_ic: Template, template_hd: float) -> Tuple[Template, Template, float]:
	'''Takes an intermediate template and produces a reference and probe with a desired HD.'''
	gspace = np.geomspace(0.2, 0.15, num=temp_ic._template.shape[0])
	hd = 0.0
	reference, probe = copy.deepcopy(temp_ic), copy.deepcopy(temp_ic)
	while not (math.isclose(hd, template_hd, abs_tol=0.005) or template_hd < hd):
		for template in (reference, probe):
			template.majority_vote()
			template.majority_vote()
			for i, row in enumerate(template._template):
				template.flip_edge(row, gspace[i])
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
	global hd_distribution
	for subject_num, (reference, probe) in enumerate(IrisCodeGenerator(len(subdirectories), hd_distribution)):
		save_dir = generated_directory / Path(str(subdirectories[subject_num]))
		reference.to_image(save_dir / Path("1.bmp"))
		probe.to_image(save_dir / Path("2.bmp"))
		reference.to_file(save_dir / Path("1.txt"))
		probe.to_file(save_dir / Path("2.txt"))

def validate(processes: int) -> None:
	'''Produces various statistics, which allow to determine whether or not the generated iris-codes have the desired statistical properties.'''
	osiris_interval = [(p.stem[:3], p.stem[-1], Template.from_image(p, None)) for p in sorted(Path("iris_codes_interval").iterdir()) if p.stem[-1] == "1"]
	osiris_biosecure = [(p, p, Template.from_image(p, None)) for p in sorted(Path("iris_codes_biosecure").iterdir()) if p.stem[-1] == "1"]
	files = list(generated_directory.glob('**/*.txt'))
	num_files = len(files)
	num_cross_comparisons = num_files * (num_files - 1) // 2
	#if num_cross_comparisons > config.validation_max_comparisons:
	#	num_files = int(math.sqrt(config.validation_max_comparisons) * 2 + 1)
	ic_sample = sorted({(path.parent, path.parent.stem, path.stem.split("_")[0]) for path in itertools.islice(files, num_files)})
	ic_sample = [(p[1], p[2], Template.from_file(p[0] / Path(p[2]+"_template.txt"), p[0] / Path(p[2]+"_mask.txt"))) for p in ic_sample]
	for t in ic_sample:
		print (validation.bit_counts(t[2]))
	#for template in ic_sample:
	#	template[2].select(8, 2)
	logging.info("Sample of %d from the produced iris-codes selected for validation" % num_files)
	validation.sequence_lengths_validation(osiris_interval, osiris_biosecure, ic_sample)
	#validation.hamming_distance_validation(ic_sample)
	logging.info("Iris-code validation complete")

if __name__ == '__main__':
	start = timer()
	set_start_method("spawn")
	logging.info("Generating iris codes")
	#logging.info("Iris-code size: %u rows, %u columns" % (config.iris_code_rows, config.iris_code_columns))
	#logging.info("Number of subjects: %u" % args.subjects)
	#logging.info("Storage directory: %s" % args.directory)
	try:
		num_pools = cpus if cpus <= cpu_count() else cpu_count
	except NotImplementedError:
		num_pools = 1
	hd_distribution = partial(weibull, 5)
	subdirectories = np.array_split(range(1, subjects+1), num_pools)
	if num_pools > 1:
		with Pool(num_pools) as p:
			p.map(produce, subdirectories)
	else:
		produce(subdirectories[0])
	#validate(cpus)
	stop = timer()
	elapsed = stop - start
	m, s = divmod(elapsed, 60)
	h, m = divmod(m, 60)
	logging.info("Time elapsed: %d:%02d:%02d" % (h, m, s))