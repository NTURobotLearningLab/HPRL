from __future__ import annotations
import math
import os

import numpy as np
from prog_policies.utils.save_file import inside_seed_save_log_file, outside_seed_save_log_file
from prog_policies.search_methods import BaseSearch
from prog_policies.base import dsl_nodes

def record_search(
    best_prog: dsl_nodes.Program,
    best_reward: float,
    search_method: BaseSearch,
    search_space,
    task_envs,
    dsl,
    output_dir_seed,
    log,
    init_time,
    output_dir,
    task,
    seed,
    program: dsl_nodes.Program = None,
) -> tuple[dsl_nodes.Program, float, int]:
    last_improve = -1
    
    if program is not None:
        progs, rewards = search_method.record_search(search_space, task_envs, init = (program, program), dsl=dsl, record_type="LLM_Search")
    else:
        progs, rewards = search_method.record_search(search_space, task_envs, dsl=dsl, record_type="Search")
    
    if rewards[-1] > best_reward:
        best_reward = rewards[-1]
        best_prog = progs[-1]
        last_improve = task_envs[0].program_num

        print(f'Task: {task}, Seed: {seed}, Program_nums: {task_envs[0].program_num}, Best Reward: {best_reward}')
        inside_seed_save_log_file(log, output_dir_seed, task_envs[0].program_num, init_time, dsl.parse_node_to_str(best_prog), best_reward, search_method.record, search_method.program_record)
        outside_seed_save_log_file(output_dir, task, seed, task_envs[0].program_num, init_time, dsl.parse_node_to_str(best_prog), best_reward, search_method.record, search_method.program_record)
    else:
        print(f'Task: {task}, Seed: {seed}, Program_nums: {task_envs[0].program_num}, Best Reward: {best_reward}', end='\r')

    return best_prog, best_reward, last_improve

def record_evaluate_program_list(
    best_prog: dsl_nodes.Program,
    best_reward: float,
    program_list: list[dsl_nodes.Program],
    search_method: BaseSearch,
    task_envs,
    dsl,
    program_reward,
    output_dir_seed,
    log,
    init_time,
    output_dir,
    task,
    seed,
    max_program_nums,
) -> tuple[dsl_nodes.Program, float, int]:
    last_improve = -1

    for p in program_list:
        reward = search_method.record_evaluate_program(p, task_envs, dsl, record_type="LLM")
        program_reward.append(reward)
        if reward > best_reward:
            best_reward = reward
            best_prog = p
            last_improve = task_envs[0].program_num

            # task_envs[0].trace_program(best_prog, os.path.join(output_dir_seed, f"{str(task_envs[0].program_num)}.gif"))
            inside_seed_save_log_file(log, output_dir_seed, task_envs[0].program_num, init_time, dsl.parse_node_to_str(best_prog), best_reward, search_method.record, search_method.program_record)
            outside_seed_save_log_file(output_dir, task, seed, task_envs[0].program_num, init_time, dsl.parse_node_to_str(best_prog), best_reward, search_method.record, search_method.program_record)

            print(f'Task: {task}, Seed: {seed}, Program_nums: {task_envs[0].program_num}, Best Reward: {best_reward}')
        else:
            print(f'Task: {task}, Seed: {seed}, Program_nums: {task_envs[0].program_num}, Best Reward: {best_reward}', end="\r")

        if best_reward >= 1 or task_envs[0].program_num >= max_program_nums:
            break

    return best_prog, best_reward, last_improve


def check_save_time(best_prog, best_reward, save_step, previous_save_program_num, search_method, task_envs, dsl, output_dir_seed, log, init_time, output_dir, task, seed):
    if task_envs[0].program_num - previous_save_program_num >= save_step:
        previous_save_program_num = task_envs[0].program_num
        
        inside_seed_save_log_file(log, output_dir_seed, task_envs[0].program_num, init_time, dsl.parse_node_to_str(best_prog), best_reward, search_method.record, search_method.program_record)
        outside_seed_save_log_file(output_dir, task, seed, task_envs[0].program_num, init_time, dsl.parse_node_to_str(best_prog), best_reward, search_method.record, search_method.program_record)
    
    return previous_save_program_num 
