from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import h5py
import os
import argparse
import progressbar
import random
import pickle
from collections import deque
import matplotlib.pyplot as plt
import sys
import numpy as np
from tqdm import tqdm

from dsl import get_DSL_option
from dsl.dsl_parse_and_trace import parse_and_trace
from util import log

import karel_option

def isOrderedSuperset(arr1, arr2):
    len1 = len(arr1)
    len2 = len(arr2)
    i = 0
    j = 0
    assert j < len2
    while i < len1 and j < len2:
        if arr1[i] == arr2[j]:
            i += 1
            j += 1
        else:
            i += 1
    if j == len2:
        return True
    return False


def generator(config):
    dir_name = config.dir_name
    h = config.height
    w = config.width
    c = len(karel_option.state_table)
    wall_prob = config.wall_prob
    marker_prob = config.marker_prob
    num_train = config.num_train
    num_test = config.num_test
    num_val = config.num_val
    num_total = num_train + num_test + num_val

    dsl = get_DSL_option(dsl_type='prob', seed=config.seed, environment='karel')
    seen_programs = set()
    count = 0
    failed_exec_count = 0
    max_demo_length_in_dataset = -1
    max_program_length_in_dataset = -1
    min_demo_length_in_dataset = float('inf')
    min_program_length_in_dataset = float('inf')

    # loading query file
    query_file_path = config.load_query_file
    query_file      = open(query_file_path, 'r')
    query_list = [query.strip() for query in query_file.readlines()]
    query_set_list = [set(q.split()) for q in query_list]
    query_result_list = [[] for q in query_list]
    print("Loaded querys:")
    for i in range(len(query_list)):
        print("query string: ", query_list[i], " , query set: ", query_set_list[i])
 
    # loading valid file_id from load_from_dir_name_check
    id_file_path_check = os.path.join(config.load_from_dir_name_check, "id.txt")
    id_file_check = open(id_file_path_check, 'r')
    id_list_checked = [p_id.strip() for p_id in id_file_check.readlines()]
    print("valid program len:", len(id_list_checked))
    
    # load files from dir 
    for file_name in os.listdir(config.load_from_dir_name):
        if file_name.endswith("hdf5"):
            f_path = os.path.join(config.load_from_dir_name, file_name)

            id_file_path = os.path.join(config.load_from_dir_name, file_name.replace("data", "id").replace("hdf5", "txt"))
            print("loading from files ...")
            print(f_path)
            print(id_file_path)

            # read file
            hdf5_file = h5py.File(f_path, 'r')
            id_file = open(id_file_path, 'r')
            id_list = id_file.readlines()
            for program_id in tqdm(id_list):
                # read program
                program_id = program_id.strip()
                
                program = hdf5_file[program_id]['program'][()]
                if program.shape[0] < 4 or program.shape[0] > config.max_program_length:
                    assert False, "Program length:{}".format(program.shape[0])
                random_code_str = dsl.intseq2str(program)
                
                # check if valid
                if program_id not in id_list_checked:
                    continue

                # drop REPEAT program
                if 'R=' in random_code_str: 
                    continue
                
                # start query matching
                #random_code_str_set = set(random_code_str.split())
                #for i in range(len(query_list)):
                #    if random_code_str_set.issuperset(query_set_list[i]):
                #        query_result_list[i].append(random_code_str)

                #for i in range(len(query_list)):
                #    if isOrderedSuperset(random_code_str.split(), query_list[i].split()):
                #        query_result_list[i].append(random_code_str)
                
                for i in range(len(query_list)):
                    if query_list[i] in random_code_str:
                        query_result_list[i].append(random_code_str)

                assert not random_code_str in seen_programs
                seen_programs.add(random_code_str)
                count += 1

            hdf5_file.close()
            id_file.close()

            print("file: {} query result: ")
            for i in range(len(query_list)):
                print("query: ", query_list[i])
                print("matched number: ", len(query_result_list[i]))


    # print loading result
    print("total loaded program count: ", count)
    print("total seen program count: ", len(seen_programs))


    # write query result to file
    query_result_file = open(os.path.join(dir_name, 'query_result.txt'), 'w')
    for i in range(len(query_list)):
        query_result_file.write('query: '+query_list[i]+" matched number: {}".format(len(query_result_list[i]))+'\n')
        for j in range(len(query_result_list[i])):
            query_result_file.write('matched result {}: '.format(j) + query_result_list[i][j]+'\n')


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir_name', type=str, default='karel_dataset_option_L20_1m_collection_analysis')
    parser.add_argument('--load_from_dir_name', type=str, default='datasets_options_L20_1m_collection/karel_dataset_option_L20_1m_collection')
    parser.add_argument('--load_query_file', type=str, default='karel_env/dataset_analysis_query.txt')
    parser.add_argument('--load_from_dir_name_check', type=str, default='datasets_options_L20_1m_merge_check_v2/karel_dataset_option_L20_1m_merge_v2')
    parser.add_argument('--height', type=int, default=8,
                        help='height of square grid world')
    parser.add_argument('--width', type=int, default=8,
                        help='width of square grid world')
    parser.add_argument('--num_train', type=int, default=800000, help='num train')
    parser.add_argument('--num_test',  type=int, default=100000,  help='num test')
    parser.add_argument('--num_val',   type=int, default=100000,  help='num val')
    parser.add_argument('--wall_prob', type=float, default=0.3,
                        help='probability of wall generation')
    parser.add_argument('--marker_prob', type=float, default=0.2,
                        help='probability of marker generation')
    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--max_program_length', type=int, default=20)
    parser.add_argument('--max_program_stmt_depth', type=int, default=6)
    parser.add_argument('--max_program_nesting_depth', type=int, default=4)
    parser.add_argument('--min_max_demo_length_for_program', type=int, default=0)
    parser.add_argument('--min_demo_length', type=int, default=0,
                        help='min demo length')
    parser.add_argument('--max_demo_length', type=int, default=50,
                        help='max demo length')
    parser.add_argument('--num_demo_per_program', type=int, default=10,
                        help='number of seen demonstrations')
    parser.add_argument('--max_demo_generation_trial', type=int, default=100)
    parser.add_argument('--cover_all_branches_in_demos', type=bool, default=False, help='cover all conditional branches while generating demonstrations')
    args = parser.parse_args()
    args.dir_name = os.path.join('datasets_options_L20_1m_collection_analysis', args.dir_name)
    check_path('datasets_options_L20_1m_collection_analysis')
    check_path(args.dir_name)

    generator(args)

if __name__ == '__main__':
    main()
