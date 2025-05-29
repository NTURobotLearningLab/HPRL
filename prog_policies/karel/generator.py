from __future__ import annotations
import numpy as np

from ..base.dsl import BaseDSL, dsl_nodes

from .environment import KarelEnvironment


class KarelStateGenerator:
    
    def __init__(self, env_args: dict, random_seed = 1) -> None:
        self.env_args = env_args
        env = KarelEnvironment(**self.env_args)
        self.state_shape = env.state_shape
        self.h = self.state_shape[1]
        self.w = self.state_shape[2]
        self.np_rng = np.random.RandomState(random_seed)
    
    def random_state(self, wall_prob=0.1, marker_prob=0.1) -> KarelEnvironment:
        s = np.zeros(self.state_shape, dtype=bool)
        # Wall
        s[4, :, :] = self.np_rng.rand(self.h, self.w) > 1 - wall_prob
        s[4, 0, :] = True
        s[4, self.h-1, :] = True
        s[4, :, 0] = True
        s[4, :, self.w-1] = True
        # Karel initial location
        valid_loc = False
        while(not valid_loc):
            y = self.np_rng.randint(0, self.h)
            x = self.np_rng.randint(0, self.w)
            if not s[4, y, x]:
                valid_loc = True
                s[self.np_rng.randint(0, 4), y, x] = True
        # Marker: num of max marker == 1 for now TODO: this is the setting for LEAPS - do we keep it?
        s[6, :, :] = (self.np_rng.rand(self.h, self.w) > 1 - marker_prob) * (s[4, :, :] == False) > 0
        s[5, :, :] = np.sum(s[6:, :, :], axis=0) == 0
        return KarelEnvironment(initial_state=s, **self.env_args)


class KarelProgramGenerator:
    
    def __init__(self, dsl: BaseDSL, random_seed = 1) -> None:
        self.dsl = dsl
        self.np_rng = np.random.RandomState(random_seed)
        self.a2i = {action: i for i, action in enumerate(self.dsl.actions + [None])}

    def fill_children_of_node(self, node: dsl_nodes.BaseNode,
                              current_depth: int = 1, current_sequence: int = 0,
                              max_depth: int = 4, max_sequence: int = 6) -> None:
        node_prod_rules = self.dsl.prod_rules[type(node)]
        for i, child_type in enumerate(node.get_children_types()):
            child_probs = self.dsl.get_dsl_nodes_probs(child_type)
            for child_type in child_probs:
                if child_type not in node_prod_rules[i]:
                    child_probs[child_type] = 0.
                if current_depth >= max_depth and child_type.get_node_depth() > 0:
                    child_probs[child_type] = 0.
            if issubclass(type(node), dsl_nodes.Concatenate) and current_sequence + 1 >= max_sequence:
                if dsl_nodes.Concatenate in child_probs:
                    child_probs[dsl_nodes.Concatenate] = 0.
            
            p_list = list(child_probs.values()) / np.sum(list(child_probs.values()))
            child = self.np_rng.choice(list(child_probs.keys()), p=p_list)
            child_instance = child()
            if child.get_number_children() > 0:
                if issubclass(type(node), dsl_nodes.Concatenate):
                    self.fill_children_of_node(child_instance, current_depth + child.get_node_depth(),
                                                current_sequence + 1, max_depth, max_sequence)
                else:
                    self.fill_children_of_node(child_instance, current_depth + child.get_node_depth(),
                                                1, max_depth, max_sequence)
            
            elif isinstance(child_instance, dsl_nodes.Action):
                child_instance.name = self.np_rng.choice(list(self.dsl.action_probs.keys()),
                                                            p=list(self.dsl.action_probs.values()))
            elif isinstance(child_instance, dsl_nodes.BoolFeature):
                child_instance.name = self.np_rng.choice(list(self.dsl.bool_feat_probs.keys()),
                                                            p=list(self.dsl.bool_feat_probs.values()))
            elif isinstance(child_instance, dsl_nodes.ConstInt):
                child_instance.value = self.np_rng.choice(list(self.dsl.const_int_probs.keys()),
                                                            p=list(self.dsl.const_int_probs.values()))
            node.children[i] = child_instance

    def generate_program(self, max_depth, max_program_size, max_sequence) -> dsl_nodes.Program:
        while True:
            program = dsl_nodes.Program()
            self.fill_children_of_node(program, max_depth=max_depth, max_sequence=max_sequence)
            if program.get_size() <= max_program_size:
                break
        return program
        
    def generate_demos(self, prog: dsl_nodes.Program, state_generator: KarelStateGenerator,
                       num_demos: int, max_demo_length: int, cover_all_branches: bool = True,
                       timeout: int = 250) -> tuple[list[list[np.ndarray]], list[list[list[bool]]], list[list[int]]]:
        action_nodes = set([n for n in prog.get_all_nodes()
                            if isinstance(n, dsl_nodes.Action)])
        n_tries = 0
        while True:
            list_bf_h = []
            list_s_h = []
            list_a_h = []
            seen_actions = set()
            while len(list_a_h) < num_demos:
                if n_tries > timeout:
                    raise Exception("Timeout while generating demos")
                env = state_generator.random_state()
                n_tries += 1
                s_h = [env.state]
                bf_h = [[bf() for bf in env.bool_features.values()]]
                a_h = []
                accepted = True
                for a in prog.run_generator(env):
                    s_h.append(env.state)
                    bf_h.append([bf() for bf in env.bool_features.values()])
                    a_h.append(self.a2i[a.name])
                    seen_actions.add(a)
                    if len(a_h) >= max_demo_length:
                        accepted = False # Reject demos that are too long
                        break
                if len(a_h) == 1 and self.np_rng.rand() < 0.8:
                    accepted = False # Reject demos that end immediately with prob 0.5
                if not accepted: continue
                list_s_h.append(s_h)
                list_bf_h.append(bf_h)
                list_a_h.append(a_h)
            if cover_all_branches and len(seen_actions) != len(action_nodes): continue
            return list_s_h, list_bf_h, list_a_h
