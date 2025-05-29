from __future__ import annotations
import copy
import numpy as np

from ..base.dsl import dsl_nodes

from .base_space import BaseSearchSpace

from ..minigrid import minigrid_node
from ..minigrid.dsl import MinigridDSL

def get_max_height(program: dsl_nodes.Program) -> int:
    """Calculates the maximum height of an input program AST

    Args:
        program (dsl_nodes.Program): Input program

    Returns:
        int: Maximum height of AST
    """
    height = 0
    for child in program.children:
        if child is not None:
            height = max(height, get_max_height(child))
    return height + program.node_depth

def get_node_current_height(node: dsl_nodes.BaseNode) -> int:
    """Calculates the current height of an input node in a program AST

    Args:
        node (dsl_nodes.BaseNode): Input node

    Returns:
        int: Current height of node
    """
    height = node.node_depth
    while not isinstance(node, dsl_nodes.Program):
        height += node.parent.node_depth
        node = node.parent
    return height

def get_max_sequence(program: dsl_nodes.Program, _current_sequence = 1, _max_sequence = 0) -> int:
    """Returns the length of maximum sequence of Concatenate nodes in an input program

    Args:
        program (dsl_nodes.Program): Input program

    Returns:
        int: Length of maximum sequence of Concatenate nodes
    """
    if isinstance(program, dsl_nodes.Concatenate):
        _current_sequence += 1
    else:
        _current_sequence = 1
    _max_sequence = max(_max_sequence, _current_sequence)
    for child in program.children:
        _max_sequence = max(_max_sequence, get_max_sequence(child, _current_sequence, _max_sequence))
    return _max_sequence

def get_node_current_sequence(node: dsl_nodes.BaseNode) -> int:
    """Returns the length of the current sequence of Concatenate nodes in an input program

    Args:
        node (dsl_nodes.BaseNode): Input node

    Returns:
        int: Length of current sequence of Concatenate nodes
    """
    current_sequence = 1
    while isinstance(node, dsl_nodes.Concatenate):
        current_sequence += 1
        node = node.parent
    return current_sequence


class ProgrammaticSpace(BaseSearchSpace):
    
    def _fill_children(self, node: dsl_nodes.BaseNode,
                          current_height: int = 1, current_sequence: int = 0,
                          max_height: int = 4, max_sequence: int = 6) -> None:
        """Recursively fills the children of a program node

        Args:
            node (dsl_nodes.BaseNode): Input node
            current_height (int, optional): Height of current element, for recursion. Defaults to 1.
            current_sequence (int, optional): Sequence of current element, for recursion. Defaults to 0.
            max_height (int, optional): Maximum allowed AST height. Defaults to 4.
            max_sequence (int, optional): Maximum allowed Concatenate sequence. Defaults to 6.
        """
        node_prod_rules = self.dsl.prod_rules[type(node)]
        for i, child_type in enumerate(node.get_children_types()):
            child_probs = self.dsl.get_dsl_nodes_probs(child_type)
            for child_type in child_probs:
                if child_type not in node_prod_rules[i]:
                    child_probs[child_type] = 0.
                if current_height >= max_height and child_type.get_node_depth() > 0:
                    child_probs[child_type] = 0.
            if isinstance(node, dsl_nodes.Concatenate) and current_sequence + 1 >= max_sequence:
                if dsl_nodes.Concatenate in child_probs:
                    child_probs[dsl_nodes.Concatenate] = 0.
            
            p_list = list(child_probs.values()) / np.sum(list(child_probs.values()))
            child = self.np_rng.choice(list(child_probs.keys()), p=p_list)
            child_instance = child()
            if child.get_number_children() > 0:
                if isinstance(node, dsl_nodes.Concatenate):
                    self._fill_children(child_instance, current_height + child.get_node_depth(),
                                        current_sequence + 1, max_height, max_sequence)
                else:
                    self._fill_children(child_instance, current_height + child.get_node_depth(),
                                        1, max_height, max_sequence)
            
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
            child_instance.parent = node

    def initialize_individual(self) -> tuple[dsl_nodes.Program, dsl_nodes.Program]:
        """Initializes individual using probabilistic DSL

        Returns:
            tuple[dsl_nodes.Program, dsl_nodes.Program]: Individual as tuple of
            program (individual) and program (decoding)
        """
        accepted = False
        while not accepted:
            program = dsl_nodes.Program()
            self._fill_children(program, max_height=4, max_sequence=6)
            prog_str = self.dsl.parse_node_to_str(program)
            accepted = get_max_height(program) <= 4 and get_max_sequence(program) <= 6 \
                and len(prog_str.split(" ")) <= 45
        return program, program
    
    def _mutate_node(self, node_to_mutate: dsl_nodes.BaseNode) -> None:
        """Mutates a node in a program by replacing it with a random node of the same type

        Args:
            node_to_mutate (dsl_nodes.BaseNode): Program node to mutate
        """
        for i, child in enumerate(node_to_mutate.parent.children):
            if child == node_to_mutate:
                child_type = node_to_mutate.parent.children_types[i]
                node_prod_rules = self.dsl.prod_rules[type(node_to_mutate.parent)]
                child_probs = self.dsl.get_dsl_nodes_probs(child_type)
                for child_type in child_probs:
                    if child_type not in node_prod_rules[i]:
                        child_probs[child_type] = 0.
                
                p_list = list(child_probs.values()) / np.sum(list(child_probs.values()))
                child = self.np_rng.choice(list(child_probs.keys()), p=p_list)
                child_instance = child()
                if child.get_number_children() > 0:
                    curr_seq = get_node_current_sequence(node_to_mutate)
                    if child_type == dsl_nodes.Concatenate:
                        curr_seq += 1
                    else:
                        curr_seq = 1
                    curr_height = get_node_current_height(node_to_mutate) + child.get_node_depth()
                    self._fill_children(child_instance, current_height=curr_height, current_sequence=curr_seq,
                        max_height=4, max_sequence=6)
                elif isinstance(child_instance, dsl_nodes.Action):
                    child_instance.name = self.np_rng.choice(list(self.dsl.action_probs.keys()),
                                                             p=list(self.dsl.action_probs.values()))
                elif isinstance(child_instance, dsl_nodes.BoolFeature):
                    child_instance.name = self.np_rng.choice(list(self.dsl.bool_feat_probs.keys()),
                                                             p=list(self.dsl.bool_feat_probs.values()))
                elif isinstance(child_instance, dsl_nodes.ConstInt):
                    child_instance.value = self.np_rng.choice(list(self.dsl.const_int_probs.keys()),
                                                              p=list(self.dsl.const_int_probs.values()))
                node_to_mutate.parent.children[i] = child_instance
                child_instance.parent = node_to_mutate.parent
            
    def get_neighbors(self, individual, k = 1) -> list[tuple[dsl_nodes.Program, dsl_nodes.Program]]:
        """Returns k neighbors of a given individual encoded as a program

        Args:
            individual (dsl_nodes.Program): Individual as a program
            k (int, optional): Number of neighbors. Defaults to 1.

        Returns:
            list[tuple[dsl_nodes.Program, dsl_nodes.Program]]: List of individuals as tuples of
            program (individual) and program (decoding)
        """
        neighbors = []
        for _ in range(k):
            # Easiest way to do a valid mutation is to do a random mutation until we find a valid one
            # This could be changed by restricting the mutation space (_fill_children args in _mutate_node)
            accepted = False
            while not accepted:
                mutated_program = copy.deepcopy(individual)
                node_to_mutate = self.np_rng.choice(mutated_program.get_all_nodes()[1:])
                self._mutate_node(node_to_mutate)
                prog_str = self.dsl.parse_node_to_str(mutated_program)
                accepted = get_max_height(mutated_program) <= 4 and get_max_sequence(mutated_program) <= 6 \
                    and len(prog_str.split(" ")) <= 45
            neighbors.append((mutated_program, mutated_program))
        return neighbors
    

class MinigridProgrammaticSpace(ProgrammaticSpace):
    def __init__(self, dsl: MinigridDSL, sigma: float = 0.25):
        super().__init__(dsl, sigma)

    def _fill_children(self, node: dsl_nodes.BaseNode,
                          current_height: int = 1, current_sequence: int = 0,
                          max_height: int = 4, max_sequence: int = 6) -> None:
        """Recursively fills the children of a program node

        Args:
            node (dsl_nodes.BaseNode): Input node
            current_height (int, optional): Height of current element, for recursion. Defaults to 1.
            current_sequence (int, optional): Sequence of current element, for recursion. Defaults to 0.
            max_height (int, optional): Maximum allowed AST height. Defaults to 4.
            max_sequence (int, optional): Maximum allowed Concatenate sequence. Defaults to 6.
        """
        # Minigrid environment is different from the general one because boolean features contains terminal node and non-terminal node
        # So, we need to handle the boolean features separately
        node_prod_rules = self.dsl.prod_rules[type(node)]
        for i, child_type in enumerate(node.get_children_types()):
            child_probs = self.dsl.get_dsl_nodes_probs(child_type)
            for child_type in child_probs:
                if child_type not in node_prod_rules[i]:
                    child_probs[child_type] = 0.
                if current_height >= max_height and child_type.get_node_depth() > 0:
                    child_probs[child_type] = 0.
            if isinstance(node, dsl_nodes.Concatenate) and current_sequence + 1 >= max_sequence:
                if dsl_nodes.Concatenate in child_probs:
                    child_probs[dsl_nodes.Concatenate] = 0.
            
            p_list = list(child_probs.values()) / np.sum(list(child_probs.values()))
            child = self.np_rng.choice(list(child_probs.keys()), p=p_list)
            child_instance = child()
            if child.get_number_children() > 0:
                if isinstance(node, dsl_nodes.Concatenate):
                    self._fill_children(child_instance, current_height + child.get_node_depth(),
                                        current_sequence + 1, max_height, max_sequence)
                else:
                    self._fill_children(child_instance, current_height + child.get_node_depth(),
                                        1, max_height, max_sequence)
            
            elif isinstance(child_instance, dsl_nodes.Action):
                child_instance.name = self.np_rng.choice(list(self.dsl.action_probs.keys()),
                                                         p=list(self.dsl.action_probs.values()))
            elif isinstance(child_instance, dsl_nodes.BoolFeature):
                child_instance.name = self.np_rng.choice(list(self.dsl.bool_feat_probs.keys()),
                                                         p=list(self.dsl.bool_feat_probs.values()))
                # case for MinigridObjectFeatureNode and MinigridColorFeatureNode
                if child_instance.name == 'front_object_type':
                    child_instance = minigrid_node.MinigridObjectFeatureNode("front_object_type")
                    child_instance.object = self.np_rng.choice(list(self.dsl.obj_feat_probs.keys()),
                                                            p=list(self.dsl.obj_feat_probs.values()))
                elif child_instance.name == 'front_object_color':
                    child_instance = minigrid_node.MinigridColorFeatureNode("front_object_color")
                    child_instance.color = self.np_rng.choice(list(self.dsl.color_feat_probs.keys()),
                                                            p=list(self.dsl.color_feat_probs.values()))
            elif isinstance(child_instance, dsl_nodes.ConstInt):
                child_instance.value = self.np_rng.choice(list(self.dsl.const_int_probs.keys()),
                                                          p=list(self.dsl.const_int_probs.values()))
            node.children[i] = child_instance
            child_instance.parent = node


    def _mutate_node(self, node_to_mutate):
        """Mutates a node in a program by replacing it with a random node of the same type

        Args:
            node_to_mutate (dsl_nodes.BaseNode): Program node to mutate
        """
        # Like the _fill_children method, we need to handle the boolean features separately
        for i, child in enumerate(node_to_mutate.parent.children):
            if child == node_to_mutate:
                child_type = node_to_mutate.parent.children_types[i]
                node_prod_rules = self.dsl.prod_rules[type(node_to_mutate.parent)]
                child_probs = self.dsl.get_dsl_nodes_probs(child_type)
                for child_type in child_probs:
                    if child_type not in node_prod_rules[i]:
                        child_probs[child_type] = 0.
                
                p_list = list(child_probs.values()) / np.sum(list(child_probs.values()))
                child = self.np_rng.choice(list(child_probs.keys()), p=p_list)
                child_instance = child()
                if child.get_number_children() > 0:
                    curr_seq = get_node_current_sequence(node_to_mutate)
                    if child_type == dsl_nodes.Concatenate:
                        curr_seq += 1
                    else:
                        curr_seq = 1
                    curr_height = get_node_current_height(node_to_mutate) + child.get_node_depth()
                    self._fill_children(child_instance, current_height=curr_height, current_sequence=curr_seq,
                        max_height=4, max_sequence=6)
                elif isinstance(child_instance, dsl_nodes.Action):
                    child_instance.name = self.np_rng.choice(list(self.dsl.action_probs.keys()),
                                                             p=list(self.dsl.action_probs.values()))
                elif isinstance(child_instance, dsl_nodes.BoolFeature):
                    child_instance.name = self.np_rng.choice(list(self.dsl.bool_feat_probs.keys()),
                                                             p=list(self.dsl.bool_feat_probs.values()))
                    # case for MinigridObjectFeatureNode and MinigridColorFeatureNode
                    if child_instance.name == 'front_object_type':
                        child_instance = minigrid_node.MinigridObjectFeatureNode('front_object_type')
                        child_instance.object = self.np_rng.choice(list(self.dsl.obj_feat_probs.keys()),
                                                                p=list(self.dsl.obj_feat_probs.values()))
                    elif child_instance.name == 'front_object_color':
                        child_instance = minigrid_node.MinigridColorFeatureNode('front_object_color')
                        child_instance.color = self.np_rng.choice(list(self.dsl.color_feat_probs.keys()),
                                                                p=list(self.dsl.color_feat_probs.values()))
                elif isinstance(child_instance, dsl_nodes.ConstInt):
                    child_instance.value = self.np_rng.choice(list(self.dsl.const_int_probs.keys()),
                                                              p=list(self.dsl.const_int_probs.values()))
                node_to_mutate.parent.children[i] = child_instance
                child_instance.parent = node_to_mutate.parent
