from __future__ import annotations

import sys
sys.path.append('.')

from prog_policies.base import dsl_nodes
from prog_policies.base import BaseDSL
from prog_policies.base.dsl_nodes import *

INDENT = "    "

def single_node_to_indent_DSL(program: BaseNode, indent: int = 0) -> str:
    output = ""
    
    if type(program) == Program:
        output += INDENT * indent + "DEF:\n"
        output += single_node_to_indent_DSL(program.children[0], indent + 1)
        return output
    
    if type(program) == While:
        output += INDENT * indent + "WHILE "
        output += single_node_to_indent_DSL(program.children[0], indent + 1) + ":\n"
        output += single_node_to_indent_DSL(program.children[1], indent + 1)
        return output
        
    if type(program) == Repeat:
        output += INDENT * indent + "REPEAT R=" + str(program.children[0].value) + ":\n"
        output += single_node_to_indent_DSL(program.children[1], indent + 1)
        return output
        
    if type(program) == If:
        output += INDENT * indent + "IF "
        output += single_node_to_indent_DSL(program.children[0], indent + 1) + ":\n"
        output += single_node_to_indent_DSL(program.children[1], indent + 1)
        return output
    
    if type(program) == ITE:
        output += INDENT * indent + "IFELSE "
        output += single_node_to_indent_DSL(program.children[0], indent + 1) + ":\n"
        output += single_node_to_indent_DSL(program.children[1], indent + 1)
        output += INDENT * indent + "ELSE:\n"
        output += single_node_to_indent_DSL(program.children[2], indent + 1)
        return output
    
    if type(program) == Concatenate:
        output += single_node_to_indent_DSL(program.children[0], indent)
        output += single_node_to_indent_DSL(program.children[1], indent)
        return output
    
    if type(program) == Not:
        output += "NOT " + single_node_to_indent_DSL(program.children[0], indent + 1)
        return output
    
    if type(program) == Action:
        output += INDENT * indent + program.name + "()" + "\n"
        return output
        
    if type(program) == BoolFeature:
        output += program.name + "()"
        return output

def single_node_to_indent_python(program: BaseNode, indent: int = 0) -> str:
    output = ""
    
    if type(program) == Program:
        output += INDENT * indent + "def run():\n"
        output += single_node_to_indent_python(program.children[0], indent + 1)
        return output
    
    if type(program) == While:
        output += INDENT * indent + "while "
        output += single_node_to_indent_python(program.children[0], indent + 1) + ":\n"
        output += single_node_to_indent_python(program.children[1], indent + 1)
        return output
        
    if type(program) == Repeat:
        output += INDENT * indent + "for i in range(" + str(program.children[0].value) + "):\n"
        output += single_node_to_indent_python(program.children[1], indent + 1)
        return output
        
    if type(program) == If:
        output += INDENT * indent + "if "
        output += single_node_to_indent_python(program.children[0], indent + 1) + ":\n"
        output += single_node_to_indent_python(program.children[1], indent + 1)
        return output
    
    if type(program) == ITE:
        output += INDENT * indent + "if "
        output += single_node_to_indent_python(program.children[0], indent + 1) + ":\n"
        output += single_node_to_indent_python(program.children[1], indent + 1)
        output += INDENT * indent + "else:\n"
        output += single_node_to_indent_python(program.children[2], indent + 1)
        return output
    
    if type(program) == Concatenate:
        output += single_node_to_indent_python(program.children[0], indent)
        output += single_node_to_indent_python(program.children[1], indent)
        return output
    
    if type(program) == Not:
        output += "not " + single_node_to_indent_python(program.children[0], indent + 1)
        return output
    
    if type(program) == Action:
        output += INDENT * indent + program.name + "()" + "\n"
        return output
        
    if type(program) == BoolFeature:
        output += program.name + "()"
        return output
    
def record_single_node_to_indent_python(program: BaseNode, indent: int = 0) -> str:
    output = ""
    
    if type(program) == Program:
        output += INDENT * indent + "def run():\n"
        output += record_single_node_to_indent_python(program.children[0], indent + 1)
        return output
    
    if type(program) == While:
        output += INDENT * indent + "while "
        output += record_single_node_to_indent_python(program.children[0], indent + 1)
        output += record_single_node_to_indent_python(program.children[1], indent + 1)
        return output
        
    if type(program) == Repeat:
        output += INDENT * indent + "for i in range(" + str(program.children[0].value) + "):\n"
        output += record_single_node_to_indent_python(program.children[1], indent + 1)
        return output
        
    if type(program) == If:
        output += INDENT * indent + "if "
        output += record_single_node_to_indent_python(program.children[0], indent + 1)
        output += record_single_node_to_indent_python(program.children[1], indent + 1)
        return output
    
    if type(program) == ITE:
        output += INDENT * indent + "if "
        output += record_single_node_to_indent_python(program.children[0], indent + 1)
        output += record_single_node_to_indent_python(program.children[1], indent + 1)
        output += INDENT * indent + "else:\n"
        output += record_single_node_to_indent_python(program.children[2], indent + 1)
        return output
    
    if type(program) == Concatenate:
        output += record_single_node_to_indent_python(program.children[0], indent)
        output += record_single_node_to_indent_python(program.children[1], indent)
        return output
    
    if type(program) == Not:
        output += "not " + record_single_node_to_indent_python(program.children[0], indent + 1)
        return output
    
    if type(program) == Action:
        if program.current:
            output += INDENT * indent + program.name + "()  # Currently executing this line" + "\n"
        else:
            output += INDENT * indent + program.name + "()" + "\n"
        return output
        
    if type(program) == BoolFeature:
        if program.current:
            output += program.name + "():  # Currently executing this line\n"
        else:
            output += program.name + "():\n"
        
        return output

        
def node_to_indent_DSL(program: dsl_nodes.Program) -> str:
    return single_node_to_indent_DSL(program, 0)

def str_to_indent_DSL(program_str: str, dsl: type[BaseDSL]) -> str:
    program = dsl.parse_str_to_node(program_str)
    return node_to_indent_DSL(program)

def node_to_indent_python(program: dsl_nodes.Program) -> str:
    return single_node_to_indent_python(program, 0)

def str_to_indent_python(program_str: str, dsl: type[BaseDSL]) -> str:
    program = dsl.parse_str_to_node(program_str)
    return node_to_indent_python(program)

def record_node_to_indent_python(program: dsl_nodes.Program) -> str:
    return record_single_node_to_indent_python(program)


# from prog_policies.karel import KarelDSL
# if __name__ == "__main__":
#     dsl = KarelDSL()
#     program_str = "DEF run m( REPEAT R=5 r( turnRight move pickMarker turnRight r) m)"
#     output = record_node_to_indent_python(dsl.parse_str_to_node(program_str))