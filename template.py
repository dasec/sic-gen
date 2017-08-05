import cv2
import itertools
import numpy as np
import os
from pathlib import Path
import random
from scipy.signal import medfilt2d
from typing import Generator, List, Tuple

def ensure_bounds(index: int, change: int, iris_code_columns: int) -> int:
	'''Makes sure that the sequence changes do not exceed row bounds.'''
	new_index = index + change
	if new_index >= iris_code_columns:
		new_index = iris_code_columns-1
	elif new_index < 0:
		new_index = 0
	return new_index

def sequence_length_generator(average_sequence_size_mu: float, average_sequence_size_sigma: float) -> Generator[int, None, None]:
	'''Generates normally distributed length of iris-code sequences around a provided mean and sigma values.'''
	while True:
		yield int(np.rint(average_sequence_size_sigma * np.random.randn() + average_sequence_size_mu))

def create_dome(mask: np.ndarray, x_position: int, span_horizontal: int, span_vertical: int, probabilitites: np.ndarray = None) -> None:
	def get_corresponding_circle(span_horizontal: float, span_vertical: float) -> Tuple[float, float]:
		'''Computes a circle corresponding to given spans.'''
		radius = span_vertical / 2 + ((span_horizontal * 2) ** 2) / (8 * span_vertical)
		x_center = mask.shape[0] - span_vertical + radius
		return radius, x_center

	def in_circle(radius: float, center_x: float, center_y: float, x: float, y: float) -> bool:
		'''Checks if a point is within a circle.'''
		c_x = center_x - x
		c_y = center_y - y
		dist_squared = c_x * c_x + c_y * c_y
		return dist_squared <= radius * radius

	def around_middle(x_position: int, span_horizontal: int, x: int) -> bool:
		'''Checks if point is around middle of the arch.'''
		cutin_size = 0.05 * np.random.randn() + 0.20
		return int(x_position - (cutin_size * span_horizontal)) < x < int(x_position + (cutin_size * span_horizontal))

	def in_bounds(x: int, y: int) -> bool:
		'''Checks if a point is within template size bounds.'''
		return 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1]

	dome_o = (*get_corresponding_circle(span_horizontal, span_vertical), x_position)
	dome1_size = np.random.randint(4,6)
	dome0_size = np.random.randint(2,4)
	dome_i = (*get_corresponding_circle(span_horizontal-dome1_size, span_vertical-dome1_size), x_position)
	dome_u = (*get_corresponding_circle(span_horizontal-dome1_size-dome0_size, span_vertical-dome1_size-dome0_size), x_position)
	all_points = list(itertools.product(range(mask.shape[0]), range(mask.shape[1])))
	circle_i = {index for index in all_points if in_circle(*dome_i, *index)}
	circle_o = {index for index in all_points if in_circle(*dome_o, *index)}
	circle_u = {index for index in all_points if in_circle(*dome_u, *index)}
	points2 = [index for index in all_points if index in circle_o]
	points1 = [index for index in all_points if index in circle_o and index not in circle_i and not around_middle(x_position, span_horizontal, index[1])]
	points0 = [index for index in all_points if index in circle_i and index not in circle_u and not around_middle(x_position, span_horizontal, index[1])]
	x, y = zip(*points1)
	mask[x, y] = np.random.random(probabilitites[x, y].shape) < probabilitites[x, y] if probabilitites is not None else 1
	x, y = zip(*points0)
	mask[x, y] = 0 if np.random.rand() > 0.5 else 1
	return points2

class Template(object):
	def __init__(self, template: np.ndarray, mask: np.ndarray = None):
		self._template = template
		self._mask = mask

	def expand(self, factor: int) -> None:
		'''Expand the template by duplicating rows.'''
		self._template = np.repeat(self._template, factor, axis=0)
		if self._mask is not None:
			self._mask = np.repeat(self._mask, factor, axis=0)

	def flip_edge(self, row: np.ndarray, flip_chance: float) -> None:
		'''Randomly flips bits at sequence edges of a row.'''
		seqs = [el for el in list(map(tuple, self.find_sequences_of_row(row, 1))) if el[1] - el[0] > 1]
		to_flip = random.sample(seqs, int(len(seqs) * flip_chance))
		starts, ends = zip(*to_flip)
		starts, ends = list(starts), list(ends)
		random.shuffle(starts)
		random.shuffle(ends)
		ls, le = len(starts), len(ends)
		sstart = [slice(0, ls // 2), slice(ls // 2 + 1, ls)]
		send = [slice(0, le // 2), slice(le // 2 + 1, le)]
		random.shuffle(sstart)
		random.shuffle(send)
		shrink_starts, grow_starts = [ensure_bounds(el, 0, row.shape[0]) for el in starts[sstart[0]]], [ensure_bounds(el, -1, row.shape[0]) for el in starts[sstart[1]]]
		shrink_ends, grow_ends = [ensure_bounds(el, -1, row.shape[0]) for el in ends[send[0]]], [ensure_bounds(el, 0, row.shape[0]) for el in ends[send[0]]]
		row[shrink_starts + shrink_ends] = 0
		row[grow_starts + grow_ends] = 1

	def find_sequences_of_all(self, value: int) -> np.ndarray:
		'''Finds consecutive sequences of given value for all rows in a template.'''
		return np.concatenate([self.find_sequences_of_row(row, value) for row in self._template])

	def find_sequences_of_row(self, row: np.ndarray, value: int) -> np.ndarray:
		'''Finds consecutive sequence of given value for a single template row.'''
		is_value = np.concatenate(([0], np.equal(row, value).view(np.uint8), [0]))
		diff = np.abs(np.diff(is_value))
		ranges = np.where(diff == 1)[0].reshape(-1, 2)
		return ranges

	def flip_range(self, row: np.ndarray, start_index: int, end_index: int) -> None:
		'''Flips the value of a range of indices in a row.'''
		row[start_index:end_index] ^= 1

	def hamming_weight(self) -> int:
		'''The Hamming weight of a template (number of 1's).'''
		return np.count_nonzero(self._template)

	def hamming_distance(self, other, rotations: int = 0, masks: bool = False, cut_rows: int = None) -> Tuple[float, int, List[float]]:
		'''Fractional Hamming distance of two iris-codes.'''
		def compute_hd(ic1: np.ndarray, ic2: np.ndarray, rotation: int, mask1: np.ndarray = None, mask2: np.ndarray = None) -> float:
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
		if masks:
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

	def initial_zigzag(self) -> None:
		'''Randomly shifts rows of a template.'''
		prev = 0
		for i in range(2, self._template.shape[0]-2, 2):
			self._template[prev:i] = np.roll(self._template[prev:i], np.random.randint(-4,4))
			prev = i

	def majority_vote(self, num_rows: int = 3, split_threshold: int = 14) -> None:
		'''Performs majority voting on a template and splits too long consecutive sequences.'''
		for i in range(0, self._template.shape[0], num_rows):
			majority = (np.sum(self._template[i:i+num_rows], axis=0) >= num_rows//2+1).astype(np.uint8, copy=False)
			seqs1 = self.find_sequences_of_row(majority, 1)
			seqs0 = self.sequences_of_0_from_sequences_of_1(seqs1)
			#if np.random.rand() > 0.5:
			self.split(majority, seqs1, split_threshold)
			#else:
			self.split(majority, seqs0, split_threshold)
			self._template[i:i+num_rows] = majority

	def medfilt2d(self) -> None:
		'''Performs a 3x3 median filtering of a template.'''
		self._template = medfilt2d(self._template).astype(np.uint8)

	def noise(self, noise_ic: np.ndarray, arch_side: str = None, noise_hd: float = None) -> None:
		'''Adds noise (eyelid arch, pupil row, random) to a template.'''
		mask = np.zeros(self._template.shape)
		arch = np.zeros(self._template.shape)
		#probabilitites = np.loadtxt("mask_probabilities.txt")
		#mask2 = (np.random.random(probabilitites.shape) < probabilitites)
		
		if arch_side:
			if arch_side == "l":
				p2 = create_dome(arch, np.random.randint(50, 150), np.random.randint(40, 80), np.random.randint(12, 24), probabilitites=None)
			else: # r
				p2 = create_dome(arch, np.random.randint(mask.shape[0]-150, mask.shape[0]-50), np.random.randint(40, 80), np.random.randint(12, 24), probabilitites=None)
			
			dome_points = set(p2)

			indices = np.nonzero(arch)
			min_x, max_x = indices[0].min(), indices[0].max()
			min_y, max_y = indices[1].min(), indices[1].max()
			x_l, y_l = max_x - min_x, max_y - min_y
			x_start = np.random.randint(0, noise_ic._template.shape[0] // 2)
			y_start = np.random.randint(0, noise_ic._template.shape[1] // 2)
			random_noise = noise_ic._template[x_start:x_start+x_l+1, y_start:y_start+y_l+1]
			arch = arch.astype(np.uint8)
			np.logical_or(random_noise, arch[min_x:min_x+x_l+1, min_y:min_y+y_l+1], out=arch[min_x:min_x+x_l+1, min_y:min_y+y_l+1], dtype=np.uint8)
			for p in np.ndindex(arch.shape):
				if p not in dome_points:
					arch[p] = 0
					
			for p in np.ndindex(self._template.shape):
				if p in dome_points:
					self._template[p] = 0
			np.logical_or(self._template, arch, out=self._template, dtype=np.uint8)

			for p in dome_points:
				mask[p[0]][p[1]] = 1
		self._mask = np.logical_not(mask).astype(np.uint8)
		if noise_hd:
			to_flip = []
			for i in range(self._template.shape[0]):
				seqs1 = self.find_sequences_of_row(self._template[i], 1)
				seq1 = list(map(tuple, seqs1))
				seq0 = list(map(tuple, self.sequences_of_0_from_sequences_of_1(seqs1)))
				seqs = sum([(seq[0] + 1, seq[1] - 1) for seq in seq0 + seq1 if seq[1] - seq[0] > 2], ())
				to_flip += [i*mask.shape[0] + el for el in np.random.choice(seqs, int((0.3 if i == 0 else noise_hd*2) * len(seqs)), replace=False)]
				self._template.flat[to_flip] ^= 1

	def remove_top_and_bottom_rows(self, n: int) -> None:
		'''Removes the top and bottom n rows of a template.'''
		self._template = self._template[n:-n]
		if self._mask is not None:
			self._mask = self._mask[n:-n]

	def select(self, every_n_row: int, every_n_column: int) -> None:
		'''Reduces template dimensions by selecting every nth row and/or column.'''
		def remove_every_nth(array, n):
			return np.array([row for i, row in enumerate(array) if i % n == 0])
		self._template = remove_every_nth(self._template, every_n_row)
		self._template = remove_every_nth(self._template.T, every_n_column).T
		if self._mask is not None:
			self._mask = remove_every_nth(self._mask, every_n_row)
			self._mask = remove_every_nth(self._mask.T, every_n_column).T

	def sequences_of_0_from_sequences_of_1(self, sequences: np.ndarray) -> np.ndarray:
		'''Given a list of sequences of 1's, computes sequences of 0's.'''
		if sequences[0][0] == 0:
			if sequences[-1][1] != self._template.shape[1]:
				result = np.append(np.ravel(sequences)[1:], self._template.shape[1]).reshape(-1,2)
			else:
				result = np.ravel(sequences)[1:-1].reshape(-1,2)
		else:
			if sequences[-1][1] != self._template.shape[1]:
				result = np.concatenate(([0], np.ravel(sequences), [self._template.shape[1]])).reshape(-1,2)
			else:
				result = np.concatenate(([0], np.ravel(sequences), [self._template.shape[1]])).reshape(-1,2)
		return result

	def set_range(self, row: np.ndarray, start_index: int, end_index: int, value: int) -> None:
		'''Sets a range of indices in a row to the given value.'''
		row[start_index:end_index] = value

	def shift(self, n: int) -> None:
		self._template = np.roll(self._template, n, axis=1)
		if self._mask is not None:
			self._mask = np.roll(self._mask, n, axis=1)

	def split(self, row: np.ndarray, sequence_indices: np.ndarray, split_threshold: int) -> None:
		'''Splits long sequences. e.g. 11111111111 -> 11110001111.'''
		start, end = zip(*sequence_indices)
		start, end = np.array(start), np.array(end)
		lengths = end - start
		indices = np.where(lengths > split_threshold)
		for start_index, end_index in sequence_indices[indices]:
			mid = int(np.rint((start_index + end_index) / 2))
			self.flip_range(row, mid-2, mid+3)

	def to_display(self) -> None:
		'''Displays a template (and mask, if present).'''
		for name, item in vars(self).items():
			if item is not None:
				cv2.imshow(name, item)
				cv2.waitKey(0)
				cv2.destroyAllWindows()

	def to_file(self, path: Path) -> None:
		'''Saves a template (and mask, if present) to text file(s).'''
		for name, item in vars(self).items():
			if item is not None:
				path.parent.mkdir(exist_ok=True)
				save_path = str(path.parent / Path(str(path.stem) + name + path.suffix))
				np.savetxt(save_path, item, fmt="%u", delimiter="", newline=os.linesep)

	def to_image(self, path: Path = None) -> None:
		'''Saves a template (and mask, if present) to image file(s).'''
		for name, item in vars(self).items():
			if item is not None:
				path.parent.mkdir(exist_ok=True)
				save_item = np.array(item, dtype=np.uint8)
				save_item[save_item == 0] = 255
				save_item[save_item == 1] = 0
				save_path = str(path.parent / Path(str(path.stem) + name + path.suffix))
				cv2.imwrite(save_path, save_item)

	@classmethod
	def create(cls, rows: int, columns: int, average_sequence_size_mu: float, average_sequence_size_sigma: float, median_filter_rows: int = 2):
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
		template = np.repeat(initial_row, rows + 2 * median_filter_rows, axis=0)
		return cls(template.astype(np.uint8), None)

	@classmethod
	def from_file(cls, template_path: Path, mask_path: Path = None):
		'''Reads a template (and optionally its mask) from text file(s).'''
		with open(template_path, 'r') as f:
			template = np.array([list(row.strip()) for row in f.readlines()], dtype=np.uint8)
		if mask_path is not None:
			with open(mask_path, 'r') as f:
				mask = np.array([list(row.strip()) for row in f.readlines()], dtype=np.uint8)
		else:
			mask = None
		return cls(template, mask)

	@classmethod
	def from_image(cls, template_path: Path, mask_path: Path = None):
		'''Reads a template (and optionally its mask) from image(s).'''
		def read_image(path):
			template = cv2.imread(str(path), 0)
			template[template == 0] = 1
			template[template == 255] = 0
			return template.astype(np.uint8)
		template = read_image(template_path)
		mask = read_image(mask_path) if mask_path is not None else None
		return cls(template, mask)
