import numpy as np
import itertools
from typing import Generator, List
import json
from scipy.signal import medfilt
import copy
import math
from pathlib import Path
import cv2
from typing import Union, Tuple

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

def from_image(path: Path) -> np.ndarray:
	'''Reads an iris-code from an image.'''
	iris_code = cv2.imread(str(path), 0)
	iris_code[iris_code == 0] = 1
	iris_code[iris_code == 255] = 0
	return iris_code

def from_file(path: Path) -> np.ndarray:
	'''Reads an iris-code from a text file.'''
	with open(path, 'r') as f:
		return np.array([list(row.strip()) for row in f.readlines()], dtype=np.int_)

def show(image: np.ndarray) -> None:
	cv2.imshow("image", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

median_filter_rows = 2
Number = Union[float, int]
noise_ic = from_image("noise_ic.bmp")
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

def add_dome(mask: np.ndarray, x_position: int, span_horizontal: int, span_vertical: int, probabilitites: np.ndarray = None) -> None:
	def get_corresponding_circle(span_horizontal: Number, span_vertical: Number) -> Tuple[Number, Number]:
		radius = span_vertical / 2 + ((span_horizontal * 2) ** 2) / (8 * span_vertical)
		x_center = 36 - span_vertical + radius
		return radius, x_center

	def in_circle(radius: Number, center_x: Number, center_y: Number, x: Number, y: Number) -> bool:
		dist_squared = (center_x - x) ** 2 + (center_y - y) ** 2
		return dist_squared <= radius ** 2

	def around_middle(x_position: int, span_horizontal: int, x: int):
		cutin_size = 0.05 * np.random.randn() + 0.20
		return int(x_position - (cutin_size * span_horizontal)) < x < int(x_position + (cutin_size * span_horizontal))

	def in_bounds(x: int, y: int):
		return 0 <= x < 36 and 0 <= y < 512
	dome_o = (*get_corresponding_circle(span_horizontal, span_vertical), x_position)
	dome1_size = np.random.randint(4,6)
	dome0_size = np.random.randint(2,4)
	#print (dome1_size, dome0_size)
	dome_i = (*get_corresponding_circle(span_horizontal-dome1_size, span_vertical-dome1_size), x_position)
	dome_u = (*get_corresponding_circle(span_horizontal-dome1_size-dome0_size, span_vertical-dome1_size-dome0_size), x_position)
	points2 = [index for index in itertools.product(range(36), range(512)) if in_circle(*dome_o, *index)]
	points1 = [index for index in itertools.product(range(36), range(512)) if in_circle(*dome_o, *index) and not in_circle(*dome_i, *index) and not around_middle(x_position, span_horizontal, index[1])]
	points0 = [index for index in itertools.product(range(36), range(512)) if in_circle(*dome_i, *index) and not in_circle(*dome_u, *index) and not around_middle(x_position, span_horizontal, index[1])]
	x, y = zip(*points1)
	mask[x, y] = np.random.random(probabilitites[x, y].shape) < probabilitites[x, y] if probabilitites is not None else 1
	x, y = zip(*points0)
	mask[x, y] = 0 if np.random.rand() > 0.5 else 1
	return points2, points1, points0

class Template(object):
	def __init__(self, array):
		self._template = array
		self._mask = None

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
		self._template = np.roll(self._template, n, axis=1)
		if self._mask is not None:
			self._mask = np.roll(self._mask, n, axis=1)

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
		self._template = np.repeat(self._template, factor, axis=0)
		self._mask = np.repeat(self._mask, factor, axis=0)

	def remove_top_and_bottom_rows(self, rows):
		self._template = self._template[rows:-rows]
		self._mask = self._mask[rows:-rows]
		
	def medfilt(self):
		self._template = medfilt(self._template).astype(np.int_)

	def hamming_distance(self, other, rotations=0, mask=False, cut_rows=None):
		'''Fractional Hamming distance of two iris-codes.'''
		def compute_hd(ic1, ic2, rotation, mask1=None, mask2=None):
			ic2_r = np.array(ic2)
			ic2_r = np.roll(ic2_r, rotation, axis=1)
			if all(m is not None for m in (mask1, mask2)):
				mask2_r = np.array(mask2)
				mask2_r = np.roll(mask2_r, rotation, axis=1)
				unoccluded = np.logical_and(mask1, mask2_r)
				hd = np.count_nonzero(np.logical_and(np.logical_xor(ic1, ic2_r), unoccluded)) / np.count_nonzero(unoccluded)
			else:
				hd = np.count_nonzero(ic1 != ic2_r) / ic1.size
			return hd

		assert self._template.size == other._template.size, (self._template.shape, other._template.shape)
		ic1, ic2 = (np.array(self._template[cut_rows:-cut_rows]), np.array(other._template[cut_rows:-cut_rows])) if cut_rows else (np.array(self._template), np.array(other._template))
		if mask:
			assert self._mask.size == other._mask.size, (self._mask.shape, other._mask.shape)
			assert self._template.size == self._mask.size and other._template.size == other._mask.size, (self._template.shape, other._template.shape, self._mask.shape, other._mask.shape)
			mask1, mask2 = (np.array(self._mask[cut_rows:-cut_rows]), np.array(other._mask[cut_rows:-cut_rows])) if cut_rows else (np.array(self._mask), np.array(other._mask))
		else:
			mask1, mask2 = None, None
		best_rotation = 0
		min_hd = hd = 1.0
		hds = []
		if rotations:
			for rotation in range(-rotations, rotations+1):
				hd = compute_hd(ic1, ic2, rotation, mask1, mask2)
				hds.append((rotation, hd))
				if hd < min_hd:
					min_hd = hd
					best_rotation = rotation
		else:
			min_hd = hd = compute_hd(ic1, ic2, 0, mask1, mask2)
			hds.append(min_hd)
		return min_hd, best_rotation, hds

	def add_noise(self, arch_side=None, noise_hd=None):
		mask = np.zeros((36, 512))
		arch = np.zeros((36, 512))
		#probabilitites = np.loadtxt("mask_probabilities.txt")
		#mask2 = (np.random.random(probabilitites.shape) < probabilitites)
		
		if arch_side:
			if arch_side == "l":
				p2, p1, p0 = add_dome(arch, np.random.randint(50, 150), np.random.randint(40, 80), np.random.randint(12, 24), probabilitites=None)
			else: # r
				p2, p1, p0 = add_dome(arch, np.random.randint(512-150, 512-50), np.random.randint(40, 80), np.random.randint(12, 24), probabilitites=None)
			
			dome_points = set(p2)

			indices = np.nonzero(arch)
			min_x, max_x = indices[0].min(), indices[0].max()
			min_y, max_y = indices[1].min(), indices[1].max()
			x_l, y_l = max_x - min_x, max_y - min_y
			x_start = np.random.randint(0, noise_ic.shape[0] // 2)
			y_start = np.random.randint(0, noise_ic.shape[1] // 2)
			random_noise = noise_ic[x_start:x_start+x_l+1, y_start:y_start+y_l+1]
			arch = arch.astype(np.int_)
			np.logical_or(random_noise, arch[min_x:min_x+x_l+1, min_y:min_y+y_l+1], out=arch[min_x:min_x+x_l+1, min_y:min_y+y_l+1], dtype=np.int_)
			for p in np.ndindex(arch.shape):
				if p not in dome_points:
					arch[p] = 0
					
			for p in np.ndindex(self._template.shape):
				if p in dome_points:
					self._template[p] = 0
			np.logical_or(self._template, arch, out=self._template, dtype=np.int_)

			for p in dome_points:
				mask[p[0]][p[1]] = 1
		self._mask = np.logical_not(mask).astype(np.int_)
		to_image(self._mask)
		if noise_hd:
			to_flip = []
			for i in range(self._template.shape[0]):
				seq0 = list(map(tuple, self.find_sequences_of_row(self._template[i], 0)))
				seq1 = list(map(tuple, self.find_sequences_of_row(self._template[i], 1)))
				seqs = sum([(seq[0] + 1, seq[1] - 1) for seq in seq0 + seq1 if seq[1] - seq[0] > 2], ())
				to_flip += [i*512 + el for el in np.random.choice(seqs, int((0.3 if i == 0 else noise_hd*2) * len(seqs)), replace=False)]
				self._template.flat[to_flip] ^= 1


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
		reference = np.repeat(initial_row, rows + 2 * median_filter_rows, axis=0)
		return cls(reference)

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

reference_generation_hd = 0.4625

temp_ic = Template.create(32, 512, 6.5, 0.25)
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
	ic.add_noise(arch_side, noise_hd if noise_hd > 0 else None)
	ic.remove_top_and_bottom_rows(median_filter_rows)
	ic.expand(2)

shift = int(np.rint(2 * np.random.randn() + 2))
print ("S:", shift)
probe.shift(shift)

print (reference.hamming_distance(probe, rotations=8))
print (reference.hamming_distance(probe, rotations=8, mask=True))
print (probe._template.shape)
to_image(temp_ic._template, "b.bmp")
to_image(reference._template, "r.bmp")
to_image(probe._template, "p.bmp")
quit()