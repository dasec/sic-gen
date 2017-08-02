import numpy as np
from scipy.signal import medfilt
from pathlib import Path
from typing import Generator
import cv2

def sequence_length_generator(average_sequence_size_mu: int, average_sequence_size_sigma: int) -> Generator[int, None, None]:
	'''Generates normally distributed length of iris-code sequences around a provided mean and sigma values.'''
	while True:
		yield int(np.rint(average_sequence_size_sigma * np.random.randn() + average_sequence_size_mu))


class Template(object):
	def __init__(self, template, mask=None):
		self._template = template
		self._mask = mask

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

	def show(self) -> None:
		for name, item in vars(self).items():
			if item is not None:
				cv2.imshow(name, item)
				cv2.waitKey(0)
				cv2.destroyAllWindows()

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
		template = np.repeat(initial_row, rows + 2 * 2, axis=0)
		return cls(template, None)

	@classmethod
	def from_image(cls, template_path: Path, mask_path: Path = None):
		'''Reads a template (and optionally its mask) from image(s).'''
		def read_image(path):
			template = cv2.imread(str(path), 0)
			template[template == 0] = 1
			template[template == 255] = 0
			return template
		template = read_image(template_path)
		mask = read_image(mask_path) if mask_path is not None else None
		return cls(template, mask)

	@classmethod
	def from_file(cls, template_path: Path, mask_path: Path = None):
		'''Reads a template (and optionally its mask) from text file(s).'''
		with open(template_path, 'r') as f:
			template = np.array([list(row.strip()) for row in f.readlines()], dtype=np.int_)
		if mask_path is not None:
			with open(mask_path, 'r') as f:
				mask = np.array([list(row.strip()) for row in f.readlines()], dtype=np.int_)
		else:
			mask = None
		return cls(template, mask)