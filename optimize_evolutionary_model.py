#!/usr/bin/env python3

import subprocess
from multiprocessing.pool import Pool
import argparse
import random
import os
import glob
from Bio import AlignIO
import pickle
import re
import logging
import statistics

def get_partial_align(alignment, indicis):
    n = len(indicis)
    sub_alignment = alignment[:, indicis[0]:indicis[0]+1]
    i = 1
    while i < n:
        sub_alignment += alignment[:, indicis[i]:indicis[i]+1]
        i += 1
    return sub_alignment

def get_random_index_groups(n, k):
    l = list(range(n))
    random.shuffle(l)
    if n/k < 20:
        k = int(max(n/20, 3))
    size, left = divmod(n, k)
    groups = [l[i:i+size] for i in range(0, size*k, size)]
    left = l[-left:]
    j = 0
    for i in left:
        groups[j].append(i)
        j+=1
        j = j%k
    return l, groups

def split_and_save_alignment(msa_path, out_dir, k = 3):
    alignment = AlignIO.read(msa_path, format="fasta")
    n = alignment.get_alignment_length()
    l, groups = get_random_index_groups(n, k)
    
    x = 0
    paths_x = [os.path.join(out_dir, "_".join([os.path.basename(msa_path), "part", str(x)])+".fasta") for x in range(k)]
    paths_not_x = [os.path.join(out_dir, "_".join([os.path.basename(msa_path), "not", str(x)])+".fasta") for x in range(k)]
    for group in groups:
        group.sort()
        others = [i for i in l if i not in group]
        sub_alignment_x = get_partial_align(alignment, group)
        sub_alignment_not_x = get_partial_align(alignment, others)
        AlignIO.write(sub_alignment_x, paths_x[x], "fasta")
        AlignIO.write(sub_alignment_not_x, paths_not_x[x], "fasta")
        x += 1
    
    return (k, paths_x, paths_not_x)

def save_run_descriptor_pickle(tree_file_path, input_msa_path, input_matrix_path, lambdas, K, param_limit, name, paths_a, paths_not_a, directory, pickle_path):
    parameters = {"tree path": tree_file_path,
                "input msa path": input_msa_path,
                "input matrix": input_matrix_path,
                "K": K,
                "param limit": param_limit,
                "lambdas": lambdas,
                "paths a": paths_a,
                "paths not a": paths_not_a,
                "n": len(paths_a),
                "name": name}
    pickle.dump(parameters, open(pickle_path, 'wb'))

def get_run_descriptor_pickle(pickle_path):
    parameters = pickle.load(open(pickle_path, 'rb'))
    return parameters

def run_command(command):
    proc = subprocess.Popen(command, shell=True) # This is non-blocking
    proc.wait()

def optimize_full_MSA(tree_file_path, input_msa_path, input_matrix_path, lambdas, K, param_limit, xtol, name, directory, pickle_path):
    K, paths_a, paths_not_a = split_and_save_alignment(input_msa_path, directory, K)
    save_run_descriptor_pickle(tree_file_path, input_msa_path, input_matrix_path, lambdas, K, param_limit, name, paths_a, paths_not_a, directory, pickle_path)
    command_format = "python generate_custom_substitution_matrix.py --name {name} -L {lamb} --xtol {x_tol} --param {param_limit} -D {run_dir} -T {tree_path} -M {msa_path} -F {freq_msa_path} -S {base_mat_path}"
    command_format += "\n"

    pool = Pool(55)#, maxtasksperchild=1)
    commands_to_run = []
    for lamb in lambdas:
        for path in paths_not_a:
            part_name = name + re.findall("_not_\d*", path)[0] + "_lamb_" + str(lamb)
            part_command = command_format.format(name = part_name, lamb = lamb, param_limit = param_limit, x_tol=xtol, run_dir = directory,\
                                        tree_path = tree_file_path, msa_path = path, freq_msa_path = input_msa_path, base_mat_path = input_matrix_path)
            commands_to_run.append(part_command)

        #And the full MSA:
        part_name = name + "_full_lamb_" + str(lamb)
        part_command = command_format.format(name = part_name, lamb = lamb, param_limit = param_limit, x_tol=xtol, run_dir = directory,\
                                    tree_path = tree_file_path, msa_path = input_msa_path, freq_msa_path = input_msa_path, base_mat_path = input_matrix_path)
        commands_to_run.append(part_command)
    
    logging.info(commands_to_run.__str__())
    logging.info("Launched all msa parts")
    pool.map(run_command, commands_to_run)
    pool.terminate()
    pool.join()
    del pool
    logging.info("Finished waiting, time to analyze")

def get_all_lambdas_map(directory):
    temp_matrices_files = os.listdir(os.path.join(directory, "matrices_temp/"))
    pairs = {}
    for path in temp_matrices_files:
        if "res_mat" in path:
            match = re.findall("_not_\d*_lamb_[0-9.]*", path)
            if len(match) > 0:
                res = match[0]
                seperators = [i for i in range(len(res)) if res[i]=='_']
                i = int(res[seperators[1]+1:seperators[2]])
                lamb = float(res[seperators[-1]+1:-1])
                full_path = os.path.join(directory, "matrices_temp", path)
                if i in pairs.keys():
                    pairs[i].append((lamb, full_path))
                else:
                    pairs[i] = [(lamb, full_path)]
    
    return pairs

def get_part_msa_path(directory, part):
    listing = glob.glob(directory + "/*part_" + str(part) + ".fasta")
    if len(listing) == 0:
        raise Exception("Could not find MSA path for part ", str(part), " in directory: ", directory)
    return listing[0]

def get_likelihood(tree_file_path, msa_file_path, matrix_file_path):
    command_list = ['raxml-ng', '--loglh', '--force', '--extra', 'thread-pin', '--threads', '1', '--msa', msa_file_path, '--model', 'PROTGTR{'+matrix_file_path+'}', '--tree', tree_file_path]
    result = subprocess.run(command_list, stdout=subprocess.PIPE) #Blocking
    output = result.stdout.decode('utf-8')
    likelihood_matches = re.findall("Final LogLikelihood: [-+]?\d*\.*\d+", output)
    if len(likelihood_matches) == 0:
        return (False, "")
    else:
        likelihood = likelihood_matches[0]
        likelihood = likelihood.replace("Final LogLikelihood: ", "")
        return (True, float(likelihood))

def choose_lambda(directory, tree_file_path, lambdas):
    all_parts_pairs_map = get_all_lambdas_map(directory)
    results = {l : [] for l in lambdas}
    
    for part in all_parts_pairs_map.keys():
        matrix_lambda_pairs = all_parts_pairs_map[part]
        part_path = get_part_msa_path(directory, part)
        for l, matrix_path in matrix_lambda_pairs:
            success, likelihood = get_likelihood(tree_file_path, part_path, matrix_path)
            if success:
                results[l].append(likelihood)
    
    max_likelihood = None
    chosen_lambda = None
    for l in lambdas:
        current_mean = statistics.mean(results[l])
        if max_likelihood is None or current_mean > max_likelihood:
            max_likelihood = current_mean
            chosen_lambda = l
    
    logging.info("Calculated all of these lambdas: " + str(results))
    logging.info("And chose this one: " + str(chosen_lambda))
    return chosen_lambda

def wrap_up_MSA(pickle_path, directory):
    parameters = get_run_descriptor_pickle(pickle_path)
    tree_file_path = parameters["tree path"]
    chosen_lambda = choose_lambda(directory, tree_file_path, parameters["lambdas"])
    logging.info("The chosen lambda is: " + str(chosen_lambda))
    print("The chosen lambda is: " + str(chosen_lambda))
    logging.info("Finished with the final matrix")

def get_argument_parser():
    parser = argparse.ArgumentParser(description="Runs a single tree optimization")
    subparsers = parser.add_subparsers(dest="command")
    parser_run = subparsers.add_parser('run', help='run all the MSA from scratch')
    parser_out = subparsers.add_parser('out', help='only parse an already-analized MSA')

    parser_run.add_argument("-T", dest="input_tree", type=str, required=True, help="The input tree path")
    parser_run.add_argument("-M", dest="input_msa", type=str, required=True, help="The input msa path")
    parser_run.add_argument("-S", dest="input_matrix", type=str, required=True, help="The input substitution matrix path")
    parser_run.add_argument("-L", dest="lambdas", action="extend", nargs="+", type=float, help="All lambdas to be tested")
    parser_run.add_argument("--param", dest="param_limit", type=int, default=1, required=False, help="Paramter limit")
    parser_run.add_argument("-k", dest="k", type=int, default=3, required=False, help="K folds")
    parser_run.add_argument("--name", dest="name", type=str, required=True, help="job name")
    parser_run.add_argument("--dir", dest="dir", required=True, help="The full path of the directory of all the files")
    parser_run.add_argument("--xtol", dest="xtol", default=60, type=float, required=False, help="Defined X tolerance of the optimization, default is 60")

    parser_out.add_argument("-d", dest="dir", required=True, help="The full path of the directory of the MSA analized data")
    parser_out.add_argument("-p", dest="pickle_path", required=True, help="The full path of the pickle file")
    return parser

def out(args):
    log_file = os.path.join(args.dir, "out_log.log")
    logging.basicConfig(filename = log_file, level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    wrap_up_MSA(args.pickle_path, args.dir)

def run(args):
    identifier = "_".join([args.name, str(round(random.random()*10000)), "K", str(args.k), "P", str(args.param_limit), "L"] + [str(l) for l in args.lambdas])
    directory = os.path.join(args.dir, identifier)
    os.mkdir(directory)
    log_file = os.path.join(directory, "log.log")
    logging.basicConfig(filename = log_file, level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    print("This optimization is preformed with the following parameters:\n",
            "tree path: \n", args.input_tree,
            "\nmsa path: \n", args.input_msa,
            "\noriginal matrix path: \n", args.input_matrix,
            "\nlambdas: \n", args.lambdas,
            "\nparams limitation: \n", args.param_limit,
            "\nxtol: \n", args.xtol,
            "\nname: \n", args.name,
            "\ndirectory: \n", directory)
    pickle_path = os.path.join(directory, args.name + ".pick")

    optimize_full_MSA(args.input_tree, args.input_msa, args.input_matrix, args.lambdas, args.k, args.param_limit, args.xtol, args.name, directory, pickle_path)
    wrap_up_MSA(pickle_path, directory)

def main(args):
    if args.command == "out":
        out(args)
    else:
        run(args)

if __name__ == "__main__":
    os.environ['NUMEXPR_MAX_THREADS'] = '128'
    parser = get_argument_parser()
    args = parser.parse_args()
    main(args)