import numpy as np
import itertools
from typing import Generator, List
import json
from scipy.signal import medfilt
import copy
import math
from pathlib import Path
import cv2

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


def sequence_length_generator(average_sequence_size_mu: int, average_sequence_size_sigma: int) -> Generator[int, None, None]:
	'''Generates normally distributed length of iris-code sequences around a provided mean and sigma values.'''
	while True:
		yield int(np.rint(average_sequence_size_sigma * np.random.randn() + average_sequence_size_mu))

def weibull(shape, m=0.0001, t_min=0.015, t_max=0.45):
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
	
class Template(object):
	def __init__(self, array):
		self._template = array

	def flip_row(self, row):
		pass

	def flip_edge(self, i, j, previous_bit, current_bit, flip_chance):
		if previous_bit == 1 and current_bit == 0:
			if np.random.rand() < flip_chance:
				if np.random.rand() < 0.5:
					self._template[i][j] = 1 # grow
				else:
					self._template[i][j-1] = 0 # shrink
		elif previous_bit == 0 and current_bit == 1:
			if np.random.rand() < flip_chance:
				if np.random.rand() < 0.5:
					self._template[i][j] = 0 # grow
				else:
					self._template[i][j-1] = 1 # shrink

	def find_sequences_of_all(self, value):
		return [self.find_sequences_of_row(row, value) for row in self._template]

	def find_sequences_of_row(self, row, value):
		is_value = np.concatenate(([0], np.equal(row, value).view(np.int8), [0]))
		diff = np.abs(np.diff(is_value))
		ranges = np.where(diff == 1)[0].reshape(-1, 2)
		return ranges

	def shift(self, n):
		self._template = np.roll(self._template, n)

	def split(self, row, sequence_indices, split_threshold):
		'''Splits long sequences. e.g. 11111111111 -> 11110001111.'''
		for start_index, end_index in sequence_indices:
			sequence_length = end_index - start_index
			if sequence_length > split_threshold:
				mid = int(np.rint((np.mean([start_index, end_index]))))
				self.flip_range(row, mid-2, mid+3)

	def flip_range(self, row, start_index, end_index):
		'''Flips the value of a range of indices in a row.'''
		row[start_index:end_index] ^= 1

	def set_range(self, row, start_index, end_index, value):
		'''Sets a range of indices in a row to the given value.'''
		row[start_index:end_index] = value

	def initial_zigzag(self):
		prev = 0
		for i in range(2, self._template.shape[0]-2, 2):
			self._template[prev:i] = np.roll(self._template[prev:i], np.random.randint(-4,4))
			prev = i

	def majority_vote(self, num_rows=3, split_threshold=14):
		for i in range(0, self._template.shape[0], num_rows):
			majority = (np.sum(self._template[i:i+num_rows], axis=0) >= num_rows//2+1).astype(np.int_, copy=False)
			seqs = self.find_sequences_of_row(majority, 1)
			self.split(majority, seqs, split_threshold)
			seqs = self.find_sequences_of_row(majority, 0)
			self.split(majority, seqs, split_threshold)
			self._template[i:i+num_rows] = majority

	def expand(self, factor):
		return np.repeat(iris_code, factor, axis=0)

	def size(self):
		return self._template.size

	def shape(self):
		return self._template.shape

	def medfilt(self):
		self._template = medfilt(self._template).astype(np.int_)

	def hamming_distance(self, other, rotations=0, cut_rows=None):
		'''Fractional Hamming distance of two iris-codes.'''
		if cut_rows:
			self._template = self._template[config.extra_rows_for_median_filter:-config.extra_rows_for_median_filter]
			other._template = other._template[config.extra_rows_for_median_filter:-config.extra_rows_for_median_filter]
		assert self.size() == other.size(), (self.shape(), other.shape())
		best_rotation = 0
		min_hd = hd = np.count_nonzero(self._template != other._template) / self.size()
		hds = []
		if rotations:
			for rotation in range(-rotations, rotations+1):
				other_rot = copy.deepcopy(other)
				other_rot.shift(rotation)
				hd = np.count_nonzero(self._template != other_rot._template) / self.size()
				hds.append((rotation, hd))
				if hd < min_hd:
					min_hd = hd
					best_rotation = rotation
		return min_hd, best_rotation, hds

	@classmethod
	def create(cls, rows, columns, average_sequence_size_mu, average_sequence_size_sigma):
		current = np.random.choice([0,1])
		parts = []
		i = 0
		sequence_length = sequence_length_generator(average_sequence_size_mu, average_sequence_size_sigma)
		while i < columns:
			length = next(sequence_length)
			parts.append(length * [current])
			i += length
			current ^= 1
		initial_row = [np.concatenate(parts)[:columns]]
		reference = np.repeat(initial_row, rows + 4, axis=0)
		return cls(reference)

reference_generation_hd = 0.4625
temp_ic = Template.create(32, 512, 6.5, 0.25)
temp_ic.shift(np.random.randint(10, 512 // 2))
temp_ic.initial_zigzag()
seqs = temp_ic.find_sequences_of_all(1)
target_hd = float(weibull(10))
exp_overlap = expected_overlap(target_hd)
i_hd = (target_hd + exp_overlap) / 2
b_hd = reference_generation_hd - i_hd

print ("T-HD:", target_hd)
print ("I-HD:", i_hd)
print ("B-HD:", b_hd)

def to_file(iris_code: np.ndarray, path: Path) -> None:
	'''Saves an iris-code to a text file.'''
	path.parent.mkdir(exist_ok=True)
	np.savetxt(str(path), iris_code, fmt="%u", delimiter="", newline=os.linesep)
	
def to_image(iris_code: np.ndarray, path: Path = None) -> None:
	'''Saves an iris-code as an image.'''
	iris_code = np.array(iris_code, dtype=np.uint8)
	iris_code[iris_code == 0] = 255
	iris_code[iris_code == 1] = 0
	if path:
		cv2.imwrite(str(path), iris_code)
	else:
		show(iris_code)

def show(image: np.ndarray) -> None:
	cv2.imshow("image", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def flip_barcode(temp_ic):
	gspace = np.geomspace(0.15, 0.05, num=temp_ic._template.shape[0])
	hd = 0.0
	barcode = copy.deepcopy(temp_ic)
	while not (math.isclose(hd, b_hd, abs_tol=0.025) or b_hd < hd):
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
		hd, _, _ = barcode.hamming_distance(test_ic, 0)
	return test_ic

temp_ic = flip_barcode(temp_ic)
#to_image(temp_ic._template)
reference, probe = copy.deepcopy(temp_ic), copy.deepcopy(temp_ic)

# probe reference flipping

quit()