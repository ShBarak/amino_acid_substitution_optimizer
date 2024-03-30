#!/usr/bin/env python3

import argparse
from datetime import datetime
from SubstitutionMatrix import SubstitutionMatrix
import random
import os
import sys
import warnings
from TreeOptimizer import TreeOptimizer
import subprocess
import estimagic as em
import numpy as np
from datetime import datetime
from statistics import mean, stdev
import re

def maximize(params):
    global optimizer

    time = datetime.timestamp(datetime.now())
    random_id = str(round(random.random()*100))
    candidate_matrix = optimizer.submatrix.generate_altered_model(params)
    path = optimizer.temp_matrices_folder + "altered_mat_" + random_id + "_" + str(time) + ".PAML"
    candidate_matrix.save_to_file(path)
    success, output = get_likelihood(optimizer.tree_file_path, optimizer.msa_file_path, path)
    if success:
        ll = output
    else:
        print(candidate_matrix,'\n\n', params)
        raise Exception(output)
    try:
        os.remove(path)
    except:
        print("file wasn't deleted", path)
    del candidate_matrix
    return ((-1)*np.sum(abs(params))*(optimizer.lamb)) + ll

def run_optimizer():
    global optimizer

    num_of_params = 190
    min_bound = optimizer.submatrix.get_min_bound()
    lower_bound = np.maximum(np.full(num_of_params, -1 * (optimizer.param_limit)), min_bound)
    print("lower bounds are:\n",lower_bound)
    algo_options = {
        "ftol": optimizer.ftol,
        "xtol": optimizer.xtol,
        }
    res = em.maximize(
        criterion=maximize,
        params=np.full(num_of_params, 0),
        algorithm="pygmo_cmaes",
        lower_bounds=lower_bound,
        upper_bounds=np.full(num_of_params, optimizer.param_limit),
        algo_options = algo_options,
    )
    print(res)
    print_params_statistics(res.params)
    resulted_matrix = optimizer.submatrix.generate_altered_model(res.params)
    parameters_string = "l_" + str(optimizer.lamb) + "_pl_" + str(optimizer.param_limit) + "_ftol_" + str(optimizer.ftol) + "_xtol_" + str(optimizer.xtol) + "_" + optimizer.name
    resulted_matrix_path = optimizer.temp_matrices_folder + "res_matrix_" + parameters_string + ".PAML"
    resulted_matrix.save_to_file(resulted_matrix_path)

def print_statistics(array, title = ""):
    print(title)
    print("Average value is: ", mean(array))
    print("Max value is: ", max(array))
    print("Min value is: ", min(array))
    print("Standard deviation value is: ", stdev(array))

def print_params_statistics(params):
    print_statistics(params, "parameters")
    print_statistics(abs(params), "absolute value of parameters")

def get_likelihood(tree_file_path, msa_file_path, matrix_file_path):
    command_list = ['raxml-ng', '--loglh', '--force', '--extra', 'thread-pin', '--threads', '1', '--msa', msa_file_path, '--model', 'PROTGTR{'+matrix_file_path+'}', '--tree', tree_file_path]
    result = subprocess.run(command_list, stdout=subprocess.PIPE) #Blocking
    output = result.stdout.decode('utf-8')
    likelihood_score_matches = re.findall("Final LogLikelihood: [-+]?\d*\.*\d+", output)
    if len(likelihood_score_matches) == 0:
        print("The following command failed:\n" + " ".join(command_list))
        return (False, output)
    else:
        likelihood = likelihood_score_matches[0]
        likelihood = likelihood.replace("Final LogLikelihood: ", "")
        return (True, float(likelihood))

def get_argument_parser():
    parser = argparse.ArgumentParser(description="Runs a single tree optimization")
    parser.add_argument("-T", dest="input_tree", type=str, required=True, help="The input tree path")
    parser.add_argument("-M", dest="input_msa", type=str, required=True, help="The input msa path")
    parser.add_argument("-F", dest="frequencies_msa", default=None, required=False, help="The amino acid frequencies msa path, default is same as input msa")
    parser.add_argument("-S", dest="input_matrix", type=str, required=True, help="The input substitution matrix path")
    parser.add_argument("-L", dest="lamb", type=float, default=1, required=False, help="Paramter lambda")
    parser.add_argument("--param", dest="param_limit", type=int, default=1, required=False, help="Paramter limit")
    parser.add_argument("--ftol", dest="ftol", type=float, default=5, required=False, help="Defined F tolerance of the optimization, default is 5")
    parser.add_argument("--xtol", dest="xtol", type=float, default=60, required=False, help="Defined X tolerance of the optimization, default is 100")
    parser.add_argument("--name", dest="name", type=str, required=True, help="job name")
    parser.add_argument("-D", dest="run_dir", required=True, help="The running directory")
    return parser

def main(args):
    start_time = datetime.now()

    dir_path = args.run_dir
    temp_matrices_folder = os.path.join(dir_path , "matrices_temp/")
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    if not os.path.exists(temp_matrices_folder):
        try:
            os.mkdir(temp_matrices_folder)
        except Exception as e:
            print("Failed to create matrix folder: ", e)
    frequencies_msa_path = args.input_msa
    if args.frequencies_msa is not None:
        frequencies_msa_path = args.frequencies_msa
    submatrix = SubstitutionMatrix.generate_substitution_matrix_from_file(args.input_matrix, frequencies_msa_path)

    log_file_path = os.path.join(dir_path, "run_log_" + args.name + "_" + str(round(random.random()*100)) + ".txt")
    sys.stdout = open(log_file_path, 'w')
    warnings.filterwarnings('ignore')

    global optimizer
    optimizer = TreeOptimizer(args.name, args.input_tree, args.input_msa, frequencies_msa_path, submatrix, temp_matrices_folder, args.lamb, args.param_limit, args.ftol, args.xtol)
    print(str(optimizer))
    run_optimizer()

    end_time = datetime.now()
    print("Total run time: ", (end_time - start_time))

if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    main(args)