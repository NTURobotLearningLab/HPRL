from __future__ import annotations
from abc import ABC
import copy

from . import dsl_nodes

def _find_close_token(token_list: list[str], character: str, start_index: int = 0) -> int:
    open_token = character + '('
    close_token = character + ')'
    assert token_list[start_index] == open_token, 'Invalid program'
    parentheses = 1
    for i, t in enumerate(token_list[start_index+1:]):
        if t == open_token:
            parentheses += 1
        elif t == close_token:
            parentheses -= 1
        if parentheses == 0:
            return i + 1 + start_index
    raise Exception('Invalid program')

class BaseDSL(ABC):

    def __init__(self, nodes_list: list[dsl_nodes.BaseNode] = None):
        self.nodes_list = nodes_list
        self.tokens_list = self.convert_nodes_to_tokens_list(self.nodes_list)
        self.t2i = {token: i for i, token in enumerate(self.tokens_list)}
        self.i2t = {i: token for i, token in enumerate(self.tokens_list)}
        self.actions = [n.name for n in self.nodes_list if isinstance(n, dsl_nodes.Action)]
        self.bool_features = [n.name for n in self.nodes_list if isinstance(n, dsl_nodes.BoolFeature)]
        self.int_features = [n.name for n in self.nodes_list if isinstance(n, dsl_nodes.IntFeature)]
    
    @property
    def prod_rules(self) -> dict[type[dsl_nodes.BaseNode], list[list[type[dsl_nodes.BaseNode]]]]:
        statements = [dsl_nodes.While, dsl_nodes.Repeat, dsl_nodes.If, dsl_nodes.ITE,
                      dsl_nodes.Concatenate, dsl_nodes.Action]
        booleans = [dsl_nodes.BoolFeature, dsl_nodes.Not, dsl_nodes.And, dsl_nodes.Or,
                    dsl_nodes.ConstBool]
        statements_without_concat = [dsl_nodes.While, dsl_nodes.Repeat, dsl_nodes.If,
                                     dsl_nodes.ITE, dsl_nodes.Action]
        booleans_without_not = [dsl_nodes.BoolFeature, dsl_nodes.And, dsl_nodes.Or,
                                dsl_nodes.ConstBool]
        return {
            dsl_nodes.Program: [statements],
            dsl_nodes.While: [booleans, statements],
            dsl_nodes.Repeat: [[dsl_nodes.ConstInt], statements],
            dsl_nodes.If: [booleans, statements],
            dsl_nodes.ITE: [booleans_without_not, statements, statements],
            dsl_nodes.Concatenate: [statements_without_concat, statements],
            dsl_nodes.Not: [booleans_without_not],
            dsl_nodes.And: [booleans, booleans],
            dsl_nodes.Or: [booleans, booleans]
        }

    def get_dsl_nodes_probs(self, node_type: type[dsl_nodes.BaseNode]) -> dict[dsl_nodes.BaseNode, float]:
        return {}
    
    @property
    def action_probs(self) -> dict[str, float]:
        return {}
    
    @property
    def bool_feat_probs(self) -> dict[str, float]:
        return {}
    
    @property
    def int_feat_probs(self) -> dict[str, float]:
        return {}
    
    @property
    def const_bool_probs(self) -> dict[bool, float]:
        return {}
    
    @property
    def const_int_probs(self) -> dict[int, float]:
        return {}
    
    def structure_only(self):
        structure_nodes = [n for n in self.nodes_list if not isinstance(n, dsl_nodes.BoolFeature)
                        and not isinstance(n, dsl_nodes.IntFeature)
                        and not isinstance(n, dsl_nodes.Action)
                        and not isinstance(n, dsl_nodes.ConstInt)
                        and not isinstance(n, dsl_nodes.ConstBool)] + [None]
        return BaseDSL(structure_nodes)
        
    def extend_dsl(self):
        extended_dsl = copy.deepcopy(self)
        extended_dsl.nodes_list.append(None)
        extended_dsl.tokens_list.append('<HOLE>')
        return extended_dsl

    def get_actions(self) -> list[dsl_nodes.BaseNode]:
        return self.actions
    
    def get_bool_features(self) -> list[dsl_nodes.BaseNode]:
        return self.bool_features
    
    def get_int_features(self) -> list[dsl_nodes.BaseNode]:
        return self.int_features

    def get_tokens(self) -> list[str]:
        return self.tokens_list
    
    # Note: the parse and convert methods use the formatting that LEAPS uses for Karel environment.
    # If you want to use a different formatting, you should override these methods.
    def convert_nodes_to_tokens_list(self, nodes_list: list[dsl_nodes.BaseNode]) -> list[str]:
        tokens_list = ['DEF', 'run', 'm(', 'm)']
        for node in nodes_list:
            if node is None:
                tokens_list += ['<HOLE>']

            if isinstance(node, dsl_nodes.ConstInt):
                tokens_list += ['R=' + str(node.value)]
            if isinstance(node, dsl_nodes.ConstBool):
                tokens_list += [str(node.value)]
            if isinstance(node, dsl_nodes.Action) \
                or isinstance(node, dsl_nodes.BoolFeature) \
                or isinstance(node, dsl_nodes.IntFeature):
                tokens_list += [node.name]

            if isinstance(node, dsl_nodes.While):
                tokens_list += ['WHILE', 'c(', 'c)', 'w(', 'w)']
            if isinstance(node, dsl_nodes.Repeat):
                tokens_list += ['REPEAT', 'r(', 'r)']
            if isinstance(node, dsl_nodes.If):
                tokens_list += ['IF', 'c(', 'c)', 'i(', 'i)']
            if isinstance(node, dsl_nodes.ITE):
                tokens_list += ['IFELSE', 'c(', 'c)', 'i(', 'i)', 'ELSE', 'e(', 'e)']
            if isinstance(node, dsl_nodes.Concatenate):
                tokens_list += []

            if isinstance(node, dsl_nodes.Not):
                tokens_list += ['not', 'c(', 'c)']
            if isinstance(node, dsl_nodes.And):
                tokens_list += ['and', 'c(', 'c)']
            if isinstance(node, dsl_nodes.Or):
                tokens_list += ['or', 'c(', 'c)']

        tokens_list += ['<pad>']
        return list(dict.fromkeys(tokens_list)) # Remove duplicates

    def parse_node_to_str(self, node: dsl_nodes.BaseNode) -> str:
        if node is None:
            return '<HOLE>'
        
        if isinstance(node, dsl_nodes.ConstInt):
            return 'R=' + str(node.value)
        if isinstance(node, dsl_nodes.ConstBool):
            return str(node.value)
        if isinstance(node, dsl_nodes.Action) \
            or isinstance(node, dsl_nodes.BoolFeature) \
            or isinstance(node, dsl_nodes.IntFeature):
            return node.name

        if isinstance(node, dsl_nodes.Program):
            m = self.parse_node_to_str(node.children[0])
            return f'DEF run m( {m} m)'

        if isinstance(node, dsl_nodes.While):
            c = self.parse_node_to_str(node.children[0])
            w = self.parse_node_to_str(node.children[1])
            return f'WHILE c( {c} c) w( {w} w)'
        if isinstance(node, dsl_nodes.Repeat):
            n = self.parse_node_to_str(node.children[0])
            r = self.parse_node_to_str(node.children[1])
            return f'REPEAT {n} r( {r} r)'
        if isinstance(node, dsl_nodes.If):
            c = self.parse_node_to_str(node.children[0])
            i = self.parse_node_to_str(node.children[1])
            return f'IF c( {c} c) i( {i} i)'
        if isinstance(node, dsl_nodes.ITE):
            c = self.parse_node_to_str(node.children[0])
            i = self.parse_node_to_str(node.children[1])
            e = self.parse_node_to_str(node.children[2])
            return f'IFELSE c( {c} c) i( {i} i) ELSE e( {e} e)'
        if isinstance(node, dsl_nodes.Concatenate):
            s1 = self.parse_node_to_str(node.children[0])
            s2 = self.parse_node_to_str(node.children[1])
            return f'{s1} {s2}'

        if isinstance(node, dsl_nodes.Not):
            c = self.parse_node_to_str(node.children[0])
            return f'not c( {c} c)'
        if isinstance(node, dsl_nodes.And):
            c1 = self.parse_node_to_str(node.children[0])
            c2 = self.parse_node_to_str(node.children[1])
            return f'and c( {c1} c) c( {c2} c)'
        if isinstance(node, dsl_nodes.Or):
            c1 = self.parse_node_to_str(node.children[0])
            c2 = self.parse_node_to_str(node.children[1])
            return f'or c( {c1} c) c( {c2} c)'
        
        raise Exception(f'Unknown node type: {type(node)}')
    
    def parse_str_list_to_node(self, prog_str_list: list[str]) -> dsl_nodes.BaseNode:
        # if len(prog_str_list) == 0:
        #     return EmptyStatement()
        
        if prog_str_list[0] in self.actions:
            if len(prog_str_list) > 1:
                s1 = dsl_nodes.Action(prog_str_list[0])
                s2 = self.parse_str_list_to_node(prog_str_list[1:])
                return dsl_nodes.Concatenate.new(s1, s2)
            return dsl_nodes.Action(prog_str_list[0])
        
        if prog_str_list[0] in self.bool_features:
            if len(prog_str_list) > 1:
                s1 = dsl_nodes.BoolFeature(prog_str_list[0])
                s2 = self.parse_str_list_to_node(prog_str_list[1:])
                return dsl_nodes.Concatenate.new(s1, s2)
            return dsl_nodes.BoolFeature(prog_str_list[0])
        
        if prog_str_list[0] in self.int_features:
            if len(prog_str_list) > 1:
                s1 = dsl_nodes.IntFeature(prog_str_list[0])
                s2 = self.parse_str_list_to_node(prog_str_list[1:])
                return dsl_nodes.Concatenate.new(s1, s2)
            return dsl_nodes.IntFeature(prog_str_list[0])
        
        if prog_str_list[0] == '<HOLE>':
            if len(prog_str_list) > 1:
                s1 = None
                s2 = self.parse_str_list_to_node(prog_str_list[1:])
                return dsl_nodes.Concatenate.new(s1, s2)
            return None
        
        if prog_str_list[0] == 'DEF':
            assert prog_str_list[1] == 'run', 'Invalid program'
            assert prog_str_list[2] == 'm(', 'Invalid program'
            assert prog_str_list[-1] == 'm)', 'Invalid program'
            m = self.parse_str_list_to_node(prog_str_list[3:-1])
            return dsl_nodes.Program.new(m)
        
        elif prog_str_list[0] == 'IF':
            c_end = _find_close_token(prog_str_list, 'c', 1)
            i_end = _find_close_token(prog_str_list, 'i', c_end+1)
            c = self.parse_str_list_to_node(prog_str_list[2:c_end])
            i = self.parse_str_list_to_node(prog_str_list[c_end+2:i_end])
            if i_end == len(prog_str_list) - 1: 
                return dsl_nodes.If.new(c, i)
            else:
                return dsl_nodes.Concatenate.new(
                    dsl_nodes.If.new(c, i), 
                    self.parse_str_list_to_node(prog_str_list[i_end+1:])
                )
        elif prog_str_list[0] == 'IFELSE':
            c_end = _find_close_token(prog_str_list, 'c', 1)
            i_end = _find_close_token(prog_str_list, 'i', c_end+1)
            assert prog_str_list[i_end+1] == 'ELSE', 'Invalid program'
            e_end = _find_close_token(prog_str_list, 'e', i_end+2)
            c = self.parse_str_list_to_node(prog_str_list[2:c_end])
            i = self.parse_str_list_to_node(prog_str_list[c_end+2:i_end])
            e = self.parse_str_list_to_node(prog_str_list[i_end+3:e_end])
            if e_end == len(prog_str_list) - 1: 
                return dsl_nodes.ITE.new(c, i, e)
            else:
                return dsl_nodes.Concatenate.new(
                    dsl_nodes.ITE.new(c, i, e),
                    self.parse_str_list_to_node(prog_str_list[e_end+1:])
                )
        elif prog_str_list[0] == 'WHILE':
            c_end = _find_close_token(prog_str_list, 'c', 1)
            w_end = _find_close_token(prog_str_list, 'w', c_end+1)
            c = self.parse_str_list_to_node(prog_str_list[2:c_end])
            w = self.parse_str_list_to_node(prog_str_list[c_end+2:w_end])
            if w_end == len(prog_str_list) - 1: 
                return dsl_nodes.While.new(c, w)
            else:
                return dsl_nodes.Concatenate.new(
                    dsl_nodes.While.new(c, w),
                    self.parse_str_list_to_node(prog_str_list[w_end+1:])
                )
        elif prog_str_list[0] == 'REPEAT':
            n = self.parse_str_list_to_node([prog_str_list[1]])
            r_end = _find_close_token(prog_str_list, 'r', 2)
            r = self.parse_str_list_to_node(prog_str_list[3:r_end])
            if r_end == len(prog_str_list) - 1: 
                return dsl_nodes.Repeat.new(n, r)
            else:
                return dsl_nodes.Concatenate.new(
                    dsl_nodes.Repeat.new(n, r),
                    self.parse_str_list_to_node(prog_str_list[r_end+1:])
                )
        
        elif prog_str_list[0] == 'not':
            assert prog_str_list[1] == 'c(', 'Invalid program'
            assert prog_str_list[-1] == 'c)', 'Invalid program'
            c = self.parse_str_list_to_node(prog_str_list[2:-1])
            return dsl_nodes.Not.new(c)
        elif prog_str_list[0] == 'and':
            c1_end = _find_close_token(prog_str_list, 'c', 1)
            assert prog_str_list[c1_end+1] == 'c(', 'Invalid program'
            assert prog_str_list[-1] == 'c)', 'Invalid program'
            c1 = self.parse_str_list_to_node(prog_str_list[2:c1_end])
            c2 = self.parse_str_list_to_node(prog_str_list[c1_end+2:-1])
            return dsl_nodes.And.new(c1, c2)
        elif prog_str_list[0] == 'or':
            c1_end = _find_close_token(prog_str_list, 'c', 1)
            assert prog_str_list[c1_end+1] == 'c(', 'Invalid program'
            assert prog_str_list[-1] == 'c)', 'Invalid program'
            c1 = self.parse_str_list_to_node(prog_str_list[2:c1_end])
            c2 = self.parse_str_list_to_node(prog_str_list[c1_end+2:-1])
            return dsl_nodes.Or.new(c1, c2)

        elif prog_str_list[0].startswith('R='):
            num = int(prog_str_list[0].replace('R=', ''))
            assert num is not None
            return dsl_nodes.ConstInt(num)
        elif prog_str_list[0] in ['True', 'False']:
            return dsl_nodes.ConstBool(prog_str_list[0] == 'True')
        else:
            raise Exception(f'Unrecognized token: {prog_str_list[0]}.')
    
    # The following methods should not be overridden even if using a different formatting logic
    def parse_str_to_node(self, prog_str: str) -> dsl_nodes.BaseNode:
        prog_str_list = prog_str.split(' ')
        return self.parse_str_list_to_node(prog_str_list)
    
    def parse_node_to_int(self, node: dsl_nodes.BaseNode) -> list[int]:
        prog_str = self.parse_node_to_str(node)
        return self.parse_str_to_int(prog_str)
    
    def parse_int_to_node(self, prog_tokens: list[int]) -> dsl_nodes.BaseNode:
        prog_str = self.parse_int_to_str(prog_tokens)
        return self.parse_str_to_node(prog_str)
    
    def parse_int_to_str(self, prog_tokens: list[int]) -> str:
        token_list = [self.i2t[i] for i in prog_tokens]
        return ' '.join(token_list)
    
    def parse_str_to_int(self, prog_str: str) -> list[int]:
        prog_str_list = prog_str.split(' ')
        return [self.t2i[i] for i in prog_str_list]
