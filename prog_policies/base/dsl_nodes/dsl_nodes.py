import copy
from typing import Union
from ..environment import BaseEnvironment
from .base_node import BaseNode


# Node types, for inheritance to other classes
# Int: integer functions/constants (int return)
# Bool: boolean functions/constants (bool return)
# Statement: expression or terminal action functions (no return)
class IntNode(BaseNode):

    def interpret(self, env: BaseEnvironment) -> int:
        raise Exception("Unimplemented method: interpret")


class BoolNode(BaseNode):

    def interpret(self, env: BaseEnvironment) -> bool:
        raise Exception("Unimplemented method: interpret")


class StatementNode(BaseNode):
    pass


# Terminal/Non-Terminal types, for inheritance to other classes
class TerminalNode(BaseNode):
    pass


class OperationNode(BaseNode):
    pass


# Constants
class ConstBool(BoolNode, TerminalNode):

    def __init__(self, value: bool = False):
        super().__init__()
        self.value = value

    def interpret(self, env: BaseEnvironment) -> bool:
        return self.value


class ConstInt(IntNode, TerminalNode):

    def __init__(self, value: int = 0):
        super().__init__()
        self.value = value

    def interpret(self, env: BaseEnvironment) -> int:
        return self.value

    def record_interpret(self, env: BaseEnvironment) -> int:
        return self.value


# Program as an arbitrary node with a single StatementNode child
class Program(BaseNode):

    node_size = 0
    node_depth = 1
    children_types = [StatementNode]

    def run(self, env: BaseEnvironment) -> None:
        assert self.is_complete(), "Incomplete Program"
        self.children[0].run(env)

    def run_generator(self, env: BaseEnvironment):
        assert self.is_complete(), "Incomplete Program"
        for node in self.get_all_nodes():
            node.reset_state()
        yield from self.children[0].run_generator(env)

    def record_run_generator(self, env: BaseEnvironment):
        assert self.is_complete(), "Incomplete Program"
        for node in self.get_all_nodes():
            node.reset_state()
        yield from self.children[0].record_run_generator(env)


# Expressions
class While(StatementNode, OperationNode):

    node_depth = 1
    children_types = [BoolNode, StatementNode]

    def reset_state(self):
        self.previous_envs: list[BaseEnvironment] = []
        self.previous_env_hashes: list[str] = []

    def run(self, env: BaseEnvironment) -> None:
        while self.children[0].interpret(env):
            # If we have seen this state previously, we're in an infinite loop
            for previous_env in self.previous_envs:
                if env == previous_env:
                    env.crash()
            self.previous_envs.append(copy.deepcopy(env))
            if env.is_crashed():
                return  # To avoid infinite loops
            self.children[1].run(env)

    def run_generator(self, env: BaseEnvironment):
        while self.children[0].interpret(env):
            # If we have seen this state previously, we're in an infinite loop
            env_hash = env.hash()
            for previous_env_hash in self.previous_env_hashes:
                if env_hash == previous_env_hash:
                    env.crash()
            self.previous_env_hashes.append(env_hash)
            if env.is_crashed():
                return  # To avoid infinite loops
            yield from self.children[1].run_generator(env)

    def record_run_generator(self, env: BaseEnvironment):
        while self.children[0].record_interpret(env):
            yield self.children[0]
            # If we have seen this state previously, we're in an infinite loop
            for previous_env in self.previous_envs:
                if env == previous_env:
                    env.crash()
            self.previous_envs.append(copy.deepcopy(env))
            if env.is_crashed():
                return  # To avoid infinite loops
            yield from self.children[1].record_run_generator(env)


class Repeat(StatementNode, OperationNode):

    node_depth = 1
    children_types = [IntNode, StatementNode]

    def run(self, env: BaseEnvironment) -> None:
        for _ in range(self.children[0].interpret(env)):
            self.children[1].run(env)

    def run_generator(self, env: BaseEnvironment):
        for _ in range(self.children[0].interpret(env)):
            yield from self.children[1].run_generator(env)

    def record_run_generator(self, env: BaseEnvironment):
        for _ in range(self.children[0].record_interpret(env)):
            yield from self.children[1].record_run_generator(env)


class If(StatementNode, OperationNode):

    node_depth = 1
    children_types = [BoolNode, StatementNode]

    def run(self, env: BaseEnvironment) -> None:
        if self.children[0].interpret(env):
            self.children[1].run(env)

    def run_generator(self, env: BaseEnvironment):
        if self.children[0].interpret(env):
            yield from self.children[1].run_generator(env)

    def record_run_generator(self, env: BaseEnvironment):
        yield self.children[0]
        if self.children[0].record_interpret(env):
            yield from self.children[1].record_run_generator(env)


class ITE(StatementNode, OperationNode):

    node_depth = 1
    children_types = [BoolNode, StatementNode, StatementNode]

    def run(self, env: BaseEnvironment) -> None:
        if self.children[0].interpret(env):
            self.children[1].run(env)
        else:
            self.children[2].run(env)

    def run_generator(self, env: BaseEnvironment):
        if self.children[0].interpret(env):
            yield from self.children[1].run_generator(env)
        else:
            yield from self.children[2].run_generator(env)

    def record_run_generator(self, env: BaseEnvironment):
        yield self.children[0]
        if self.children[0].record_interpret(env):
            yield from self.children[1].record_run_generator(env)
        else:
            yield from self.children[2].record_run_generator(env)


class Concatenate(StatementNode, OperationNode):

    node_size = 0
    children_types = [StatementNode, StatementNode]

    def run(self, env: BaseEnvironment) -> None:
        self.children[0].run(env)
        self.children[1].run(env)

    def run_generator(self, env: BaseEnvironment):
        yield from self.children[0].run_generator(env)
        yield from self.children[1].run_generator(env)

    def record_run_generator(self, env: BaseEnvironment):
        yield from self.children[0].record_run_generator(env)
        yield from self.children[1].record_run_generator(env)


# Boolean operations
class Not(BoolNode, OperationNode):

    children_types = [BoolNode]

    def interpret(self, env: BaseEnvironment) -> bool:
        return not self.children[0].interpret(env)

    def record_interpret(self, env: BaseEnvironment) -> bool:
        return not self.children[0].record_interpret(env)


# Note: And and Or are defined here but are not used in Karel
class And(BoolNode, OperationNode):

    children_types = [BoolNode, BoolNode]

    def interpret(self, env: BaseEnvironment) -> bool:
        return self.children[0].interpret(env) and self.children[1].interpret(env)


class Or(BoolNode, OperationNode):

    children_types = [BoolNode, BoolNode]

    def interpret(self, env: BaseEnvironment) -> bool:
        return self.children[0].interpret(env) or self.children[1].interpret(env)


# For actions available in environment
class Action(StatementNode, TerminalNode):

    def run(self, env: BaseEnvironment) -> None:
        if not env.is_crashed():
            env.run_action(self.name)

    def run_generator(self, env: BaseEnvironment):
        if not env.is_crashed():
            env.run_action(self.name)
            yield self

    def record_run_generator(self, env: BaseEnvironment):
        if not env.is_crashed():
            env.run_action(self.name)
            yield self


# For features available in environment
class BoolFeature(BoolNode, TerminalNode):

    def interpret(self, env: BaseEnvironment) -> bool:
        return env.get_bool_feature(self.name)

    def record_interpret(self, env: BaseEnvironment) -> bool:
        return env.get_bool_feature(self.name)


class IntFeature(BoolNode, TerminalNode):

    def interpret(self, env: BaseEnvironment) -> int:
        return env.get_int_feature(self.name)
