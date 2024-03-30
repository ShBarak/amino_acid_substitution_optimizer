#!/usr/bin/env python3

import os
import numpy as np
from Bio import AlignIO
from Matrix import Matrix

class SubstitutionMatrix:

	ORDER = "A R N D C Q E G H I L K M F P S T W Y V"
	MATRIX_SCALE = 1000.0

	def __init__(self, matrix, frequencies, total_sum = None):
		self.n = 20
		if type(matrix) == Matrix:
			self.matrix = matrix
		else:
			self.matrix = Matrix(matrix, SubstitutionMatrix.ORDER)
		self.frequencies = frequencies
		if total_sum is None:
			self.original_matrix_sum = self.matrix.get_total_sum()
		else:
			self.original_matrix_sum = total_sum

		#Scale for a faster optimization process
		self.scale_for_use()

	@classmethod
	def generate_substitution_matrix_from_file(cls, file_path, frequencies_msa_path):
		if not os.path.exists(file_path):
			raise Exception("File doesn't exist")
		with open(file_path) as file:
			file_content = file.read()
		split = file_content.split("\n\n")
		matrix = split[0]
		matrix, n = SubstitutionMatrix.create_matrix(matrix)
		frequencies = SubstitutionMatrix.create_frequencies_from_msa(frequencies_msa_path)
		obj = cls(matrix, frequencies)
		return obj

	@classmethod
	def create_matrix(cls, matrix_text):
		matrix_lines = [item for item in matrix_text.split("\n") if item != ""]
		n = len(SubstitutionMatrix.ORDER.split(" "))
		table = np.zeros((n,n))
		Li = 1
		for line in matrix_lines:
			row = [item for item in line.split(" ") if item != ""]
			Ci = 0
			for item in row:
				table[(Li, Ci)] = float(item)
				Ci += 1
			Li += 1
		return table, n

	@classmethod
	def create_frequencies_from_msa(cls, msa_path):
		msa = AlignIO.read(msa_path, format="fasta")
		frequencies = {}
		for aa in SubstitutionMatrix.ORDER.split(" "):
			frequencies[aa] = 0
		for sequence in msa:
			full_sequence = str(sequence.seq)
			for char in full_sequence:
				if char in SubstitutionMatrix.ORDER.replace(" ",""):
					frequencies[char.upper()] += 1
		total_sum = sum([frequencies[i] for i in frequencies.keys()])
		normalized_frequencies = {i:frequencies[i]/total_sum for i in frequencies.keys()}
		return normalized_frequencies
	
	@classmethod
	def create_frequencies_from_text(cls, freq_text):
		frequencies_list = [item for item in freq_text.split(" ") if item != ""]
		f = len(frequencies_list)
		order = SubstitutionMatrix.ORDER.split(" ")
		frequencies = {}
		for i in range(f):
			frequencies[order[i]] = float(frequencies_list[i])
		return frequencies
	
	def get_ordered_frequencies(self):
		freq = []
		order = SubstitutionMatrix.ORDER.split(" ")
		for aa in order:
			freq.append(self.frequencies[aa])
		return np.array(freq)
	
	def scale_for_use(self):
		self.matrix.multiply_matrix(self.MATRIX_SCALE / self.matrix.get_total_sum())

	def descale(self):
		self.matrix.multiply_matrix(self.original_matrix_sum / self.matrix.get_total_sum())
	
	def get_normalize_frequencies(self):
		return self.frequencies

	def get_frequency(self, i):
		return self.frequencies[i]
	
	def get_min_bound(self):
		return np.multiply(np.multiply(self.matrix.deconstract_values(), -1), 0.95)

	def generate_altered_model(self, params):
		altered_matrix = self.matrix.generate_altered_matrix(params)
		return SubstitutionMatrix(altered_matrix, self.frequencies, total_sum = self.original_matrix_sum)

	def get_matrix_text(self):
		lines = SubstitutionMatrix.ORDER.split(" ")[1:]
		cols = SubstitutionMatrix.ORDER.split(" ")[:-1]
		table_rows = []
		counter = 1
		for line in lines:
			row = [self.get_substitution(line, col) for col in cols[:counter]]
			table_rows.append(' '.join([str(item) for item in row]))
			counter += 1
		return '\n'.join(table_rows)

	def get_frequencies_text(self):
		frequencies = self.get_ordered_frequencies()
		text = ' '.join([str(f) for f in frequencies])
		return text

	def save_to_file(self, file_path):
		self.descale()
		text = "\n" + self.get_matrix_text() + "\n\n" + self.get_frequencies_text()
		with open(file_path, 'w') as file:
			file.write(text)
		self.scale_for_use()
	
	def get_substitution_matrix(self):
		return self.matrix.get_matrix()
	
	def get_substitutions_values_vector(self):
		return self.matrix.deconstract_values()

	def get_substitution(self, a, b):
		return self.matrix.get_item_by_ab(a, b)

	def get_substitution_by_index(self, index):
		return self.matrix.get_item_by_index(index)
	
	def get_original_matrix_sum(self):
		return self.original_matrix_sum
	
	def __str__(self):
		full_string = ""
		full_string += "Substitution matrix:\n\n" + str(self.matrix)
		full_string += "\n\nFrequencies:\n\n" + str(self.frequencies)
		return full_string