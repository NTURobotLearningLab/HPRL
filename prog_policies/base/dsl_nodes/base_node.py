from __future__ import annotations
from typing import Generator, Union

from ..environment import BaseEnvironment

class BaseNode:

    node_size: int = 1
    node_depth: int = 0
    children_types: list[type[BaseNode]] = []

    def __init__(self, name: Union[str, None] = None):
        self.children: list[Union[BaseNode, None]] = [None for _ in range(self.get_number_children())]
        self.parent: Union[BaseNode, None] = None
        self.value: Union[None, bool, int] = None
        self.current = False
        if name is not None:
            self.name = name
        else:
            self.name = type(self).__name__
    
    # Some nodes might use an internal state to keep track of program execution. This function
    # will be called by the program when running in a new environment
    def reset_state(self):
        pass
    
    # In this implementation, get_size is run recursively in a program, so we do not need to worry
    # about updating each node size as we grow them
    def get_size(self) -> int:
        size = self.node_size
        for child in self.children:
            if child is not None:
                size += child.get_size()
        return size
    
    # recursively calculate the node depth (number of levels from root)
    def get_depth(self) -> int:
        depth = 0
        for child in self.children:
            if child is not None:
                depth = max(depth, child.get_depth())
        return depth + self.node_depth
    
    # Recursively get all nodes in the tree
    def get_all_nodes(self) -> list[BaseNode]:
        nodes = [self]
        for child in self.children:
            if child is not None:
                nodes += child.get_all_nodes()
        return nodes
    
    def is_complete(self) -> bool:
        for child in self.children:
            if child is None:
                return False
            elif not child.is_complete():
                return False
        return True
    
    @classmethod
    def get_number_children(cls) -> int:
        return len(cls.children_types)

    @classmethod
    def get_children_types(cls) -> list[type[BaseNode]]:
        return cls.children_types
    
    @classmethod
    def get_node_size(cls) -> int:
        return cls.node_size
    
    @classmethod
    def get_node_depth(cls) -> int:
        return cls.node_depth
    
    @classmethod
    def new(cls, *args) -> BaseNode:
        inst = cls()
        children_types = cls.get_children_types()
        for i, arg in enumerate(args):
            if arg is not None:
                assert issubclass(type(arg), children_types[i])
                inst.children[i] = arg
                arg.parent = inst
        return inst
    
    # interpret is used by nodes that return a value (IntNode, BoolNode)
    def interpret(self, env: BaseEnvironment) -> Union[bool, int]:
        raise Exception('Unimplemented method: interpret')

    # run and run_generator are used by nodes that affect env (StatementNode)
    def run(self, env: BaseEnvironment) -> None:
        raise Exception('Unimplemented method: run')

    def run_generator(self, env: BaseEnvironment) -> Generator[BaseNode, None, None]:
        raise Exception('Unimplemented method: run_generator')
    
    # For record
    def record_interpret(self, env: BaseEnvironment) -> Union[bool, int]:
        raise Exception('Unimplemented method: interpret')

    def record_run_generator(self, env: BaseEnvironment) -> Generator[BaseNode, None, None]:
        raise Exception('Unimplemented method: run_generator')