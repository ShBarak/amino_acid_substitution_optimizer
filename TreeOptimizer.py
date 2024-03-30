#!/usr/bin/env python3

class TreeOptimizer:

	def __init__(self, name, tree_file_path, msa_file_path, frequencies_msa_path, submatrix, temp_matrices_folder, lamb, param_limit, ftol = 5, xtol = 100, test = False):
		self.name = name
		self.tree_file_path = tree_file_path
		self.msa_file_path = msa_file_path
		self.frequencies_msa_path = frequencies_msa_path
		self.submatrix = submatrix
		self.lamb = lamb
		self.param_limit = param_limit
		self.ftol = ftol
		self.xtol = xtol
		self.temp_matrices_folder = temp_matrices_folder
	
	def __str__(self):
		s = "Optimizer: " + str(self.name) +\
            "\ntree path: " + str(self.tree_file_path) +\
            "\nmsa path: " + str(self.msa_file_path) +\
			"\nfrequencies msa path: " + str(self.frequencies_msa_path) +\
            "\nlambdas: " + str(self.lamb) +\
            "\nparams limitation: " + str(self.param_limit) +\
            "\nftol: " + str(self.ftol) +\
			"\nxtol: " + str(self.xtol)
		return s