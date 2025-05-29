import json
import time
import copy
import os
from typing import List


def inside_seed_save_log_file(
    log: dict,
    output_dir_seed: str,
    num: int,
    init_time: float,
    best_program: str,
    best_reward: float,
    record: dict,
    program_record: dict,
) -> None:
    new_log = copy.deepcopy(log)

    new_log["num"] = num
    new_log["time"] = time.time() - init_time
    new_log["best_program"] = best_program
    new_log["best_reward"] = best_reward
    new_log["record"] = record
    new_log["program_record"] = program_record

    with open(os.path.join(output_dir_seed, f"log.json"), "w") as f:
        json.dump(new_log, f, indent=4)
        
    print(
        f"Save the inside seed log file at {output_dir_seed}, program num {num}, best reward {best_reward}"
    )


def outside_seed_save_log_file(
    output_dir: str,
    task: str,
    seed: int,
    num: int,
    init_time: float,
    best_program: str,
    best_reward: float,
    record: dict,
    program_record: dict,
) -> None:
    
    content = None
    
    for _ in range(5):
        try:
            with open(os.path.join(output_dir, f"{task}_record.json"), "r") as f:
                content = json.load(f)
        except:
            time.sleep(1)
            content = {}

    content[task] = content.get(task, {})
    content[task][str(seed)] = {}
    content[task][str(seed)]["num"] = num
    content[task][str(seed)]["time"] = time.time() - init_time
    content[task][str(seed)]["best_program"] = best_program
    content[task][str(seed)]["best_reward"] = best_reward
    content[task][str(seed)]["record"] = record
    content[task][str(seed)]["program_record"] = program_record

    with open(os.path.join(output_dir, f"{task}_record.json"), "w") as f:
        json.dump(content, f, indent=4)

    print(
        f"Save the outside seed log file at {output_dir}, program num {num}, best reward {best_reward}"
    )

def revision_inside_seed_save_log_file(
    log: dict,
    output_dir_seed: str,
    best_reward: float,
) -> None:
    new_log = copy.deepcopy(log)

    with open(os.path.join(output_dir_seed, f"log.json"), "w") as f:
        json.dump(new_log, f, indent=4)
        
    print(
        f"Save the inside seed log file at {output_dir_seed}, best reward {best_reward}"
    )


def revision_outside_seed_save_log_file(
    output_dir: str,
    task: str,
    seed: int,
    record_program_str_list: List[str],
    record_reward_list: List[float],
    best_reward: float,
) -> None:
    
    content = None
    
    for _ in range(5):
        try:
            with open(os.path.join(output_dir, f"{task}_record.json"), "r") as f:
                content = json.load(f)
        except:
            time.sleep(1)
            content = {}

    content[task] = content.get(task, {})
    content[task][str(seed)] = {}
    content[task][str(seed)]["program"] = record_program_str_list
    content[task][str(seed)]["reward"] = record_reward_list

    with open(os.path.join(output_dir, f"{task}_record.json"), "w") as f:
        json.dump(content, f, indent=4)

    print(
        f"Save the outside seed log file at {output_dir}, best reward {best_reward}"
    )