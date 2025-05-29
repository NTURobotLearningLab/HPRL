from __future__ import annotations
from typing import Union
from abc import ABC, abstractmethod
import copy

import numpy as np

from .environment import BaseEnvironment
from . import dsl_nodes
from PIL import Image

from ..utils.indent import node_to_indent_python, record_node_to_indent_python

class TerminateException(Exception):
    pass

class BaseTask(ABC):
    
    def __init__(self, env_args: dict = {}, seed: Union[int, None] = None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState()
        self.seed = seed
        self.crash_penalty = -1.
        self.initial_environment = self.generate_initial_environment(env_args)
        self.reset_environment()
        self.program_num = 0
    
    def get_environment(self) -> BaseEnvironment:
        return self.environment
    
    def reset_environment(self) -> None:
        if self.seed is not None:
            self.rng = np.random.RandomState(self.seed)
        else:
            self.rng = np.random.RandomState()
        self.environment = copy.deepcopy(self.initial_environment)
    
    @abstractmethod
    def generate_initial_environment(self, env_args: dict) -> BaseEnvironment:
        pass
    
    @abstractmethod
    def get_reward(self, environment: BaseEnvironment) -> tuple[bool, float]:
        pass
    
    def find_indices_of_ast_path(self, subprogram: dsl_nodes.BaseNode, node: dsl_nodes.BaseNode,
                                 current_index: int = 0) -> list[int]:
        # TODO: optimize -- would be better to have the node's parent as an attribute of the node (double linked)
        if subprogram == node:
            return [current_index]
        for i, child in enumerate(subprogram.children):
            child_indices = self.find_indices_of_ast_path(child, node, current_index + 1 + i)
            if len(child_indices) > 0:
                return [current_index] + child_indices
        return []
    
    def evaluate_and_assign_credit(self, program: dsl_nodes.Program) -> tuple[float, dict, dict]:
        self.reset_environment()
        reward = 0.
        all_nodes = program.get_all_nodes()
        node_score = {node: 0. for node in all_nodes}
        node_count = {node: 0 for node in all_nodes}
        action_nodes = [node for node in all_nodes if isinstance(node, dsl_nodes.Action)]
        action_scores = {node: 0. for node in action_nodes}
        action_counts = {node: 0 for node in action_nodes}
        for action in program.run_generator(self.environment):
            terminated, instant_reward = self.get_reward(self.environment)
            if self.environment.is_crashed():
                instant_reward += self.crash_penalty
            reward += instant_reward
            action_scores[action] += instant_reward
            action_counts[action] += 1
            if terminated or self.environment.is_crashed():
                break
        for action in action_nodes:
            current_node = action
            while not issubclass(type(current_node), dsl_nodes.Program):
                node_score[current_node] += action_scores[action]
                node_count[current_node] += action_counts[action]
                current_node = current_node.parent
        return reward, node_score, node_count
    
    def evaluate_program(self, program: dsl_nodes.Program) -> float:
        self.program_num += 1
        self.reset_environment()
        reward = 0.
        for _ in program.run_generator(self.environment):
            terminated, instant_reward = self.get_reward(self.environment)
            reward += instant_reward
            if terminated or self.environment.is_crashed():
                break
        return reward

    def record_evaluate_program(self, program: dsl_nodes.Program) -> tuple[float, list[list[dict[str, str]]]]:
        self.program_num += 1
        self.reset_environment()
        reward = 0.
        logs = [{"state": self.environment.to_string_worldcoder_style(), "program_str": node_to_indent_python(program)}]
        step = 0
        for node in program.record_run_generator(self.environment):
            step += 1
            if type(node) == dsl_nodes.Action:
                terminated, instant_reward = self.get_reward(self.environment)
                state = self.environment.record_partial_state()
                node.current = True
                program_str = record_node_to_indent_python(program)
                node.current = False
                reward += instant_reward
                logs.append({"state": state, "program_str": program_str, "instant_reward": instant_reward, "terminated": terminated, "name": node.name, "type": "action"})
                if terminated or self.environment.is_crashed():
                    break
            else:
                if node.name == "Not":
                    continue
                state = self.environment.record_partial_state()
                node.current = True
                program_str = record_node_to_indent_python(program)
                node.current = False
                logs.append({"state": state, "program_str": program_str, "name": node.name, "result": self.environment.get_bool_feature(node.name), "type": "perception"})
        return reward, logs

    def trace_program(self, program: dsl_nodes.Program, image_name: str = 'trace.gif', max_steps: int = 1000, save: bool = True, root_dir = "./") -> list[Image.Image]:
        
        self.reset_environment()
        im = Image.fromarray(self.environment.to_image(root_dir=root_dir))
        im_list = []
        for _ in program.run_generator(self.environment):
            terminated, _ = self.get_reward(self.environment)
            im_list.append(Image.fromarray(self.environment.to_image(root_dir=root_dir)))
            if len(im_list) > max_steps or terminated or self.environment.is_crashed():
                break
        if save:
            im.save(image_name, save_all=True, append_images=im_list, duration=75, loop=0)
            
        return im_list
