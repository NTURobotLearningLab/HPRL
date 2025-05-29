from typing import Union

from prog_policies.base import BaseDSL, dsl_nodes
from prog_policies.base.dsl import _find_close_token

from .minigrid_node import MinigridColorFeatureNode, MinigridObjectFeatureNode


class MinigridDSL(BaseDSL):

    def __init__(self):
        nodes_list = [
            dsl_nodes.Repeat(),
            dsl_nodes.Concatenate(),
            dsl_nodes.Action("left"),
            dsl_nodes.Action("right"),
            dsl_nodes.Action("forward"),
            dsl_nodes.Action("pickup"),
            dsl_nodes.Action("drop"),
            dsl_nodes.Action("toggle"),
            dsl_nodes.BoolFeature("front_is_clear"),
            MinigridColorFeatureNode.new("front_object_type", "red"),
            MinigridColorFeatureNode.new("front_object_type", "blue"),
            MinigridObjectFeatureNode.new("front_object_color", "lava"),
            MinigridObjectFeatureNode.new("front_object_color", "door"),
            MinigridObjectFeatureNode.new("front_object_color", "ball"),
            MinigridObjectFeatureNode.new("front_object_color", "box"),
            dsl_nodes.BoolFeature("is_carrying_object"),
        ] + [dsl_nodes.ConstInt(i) for i in range(20)]
        super().__init__(nodes_list)

    @property
    def prod_rules(self):
        statements = [
            dsl_nodes.While,
            dsl_nodes.Repeat,
            dsl_nodes.If,
            dsl_nodes.ITE,
            dsl_nodes.Concatenate,
            dsl_nodes.Action,
        ]
        booleans = [
            dsl_nodes.BoolFeature,
            dsl_nodes.Not,
            MinigridColorFeatureNode,
            MinigridObjectFeatureNode,
        ]
        statements_without_concat = [
            dsl_nodes.While,
            dsl_nodes.Repeat,
            dsl_nodes.If,
            dsl_nodes.ITE,
            dsl_nodes.Action,
        ]
        booleans_without_not = [
            dsl_nodes.BoolFeature,
            MinigridColorFeatureNode,
            MinigridObjectFeatureNode,
        ]

        return {
            dsl_nodes.Program: [statements],
            dsl_nodes.While: [booleans, statements],
            dsl_nodes.Repeat: [[dsl_nodes.ConstInt], statements],
            dsl_nodes.If: [booleans, statements],
            dsl_nodes.ITE: [booleans_without_not, statements, statements],
            dsl_nodes.Concatenate: [statements_without_concat, statements],
            dsl_nodes.Not: [booleans_without_not],
        }

    def get_dsl_nodes_probs(
        self, node_type
    ) -> dict[Union[str, dsl_nodes.BaseNode], float]:
        if node_type == dsl_nodes.StatementNode:
            return {
                dsl_nodes.While: 0.15,
                dsl_nodes.Repeat: 0.03,
                dsl_nodes.Concatenate: 0.5,
                dsl_nodes.If: 0.08,
                dsl_nodes.ITE: 0.04,
                dsl_nodes.Action: 0.2,
            }
        elif node_type == dsl_nodes.BoolNode:
            return {dsl_nodes.BoolFeature: 0.5, dsl_nodes.Not: 0.5}
        elif node_type == dsl_nodes.BoolFeature:
            return {
                "front_is_clear": 0.125,
                MinigridObjectFeatureNode: 0.5,
                MinigridColorFeatureNode: 0.25,
                "is_carrying_object": 0.125,
            }
        elif node_type == dsl_nodes.IntNode:
            return {dsl_nodes.ConstInt: 1.0}
        else:
            raise ValueError(f"Unknown node type: {node_type}")

    @property
    def action_probs(self):
        return {
            "left": 0.15,
            "right": 0.15,
            "forward": 0.5,
            "pickup": 0.08,
            "drop": 0.08,
            "toggle": 0.04,
        }

    @property
    def bool_feat_probs(self):
        return {
            "front_is_clear": 0.125,
            "front_object_type": 0.5,
            "front_object_color": 0.25,
            "is_carrying_object": 0.125,
        }

    @property
    def obj_feat_probs(self):
        return {"lava": 0.25, "door": 0.25, "ball": 0.25, "box": 0.25}

    @property
    def color_feat_probs(self):
        return {"red": 0.5, "blue": 0.5}

    @property
    def const_int_probs(self):
        return {i: 1 / 20 for i in range(20)}

    def convert_nodes_to_tokens_list(self, nodes_list):
        tokens_list = super().convert_nodes_to_tokens_list(nodes_list)
        for node in nodes_list:
            if isinstance(node, MinigridObjectFeatureNode):
                tokens_list += [node.name, "h(", node.object, "h)"]
            elif isinstance(node, MinigridColorFeatureNode):
                tokens_list += [node.name, "h(", node.color, "h)"]
        tokens_list = list(dict.fromkeys(tokens_list))
        return tokens_list

    def parse_node_to_str(self, node: dsl_nodes.BaseNode) -> str:
        if node is None:
            return "<HOLE>"

        if isinstance(node, dsl_nodes.ConstInt):
            return "R=" + str(node.value)
        if isinstance(node, dsl_nodes.ConstBool):
            return str(node.value)
        if isinstance(node, dsl_nodes.Action) or isinstance(node, dsl_nodes.IntFeature):
            return node.name

        # boolean features
        if isinstance(node, MinigridObjectFeatureNode):
            return f"{node.name} h( {node.object} h)"
        if isinstance(node, MinigridColorFeatureNode):
            return f"{node.name} h( {node.color} h)"
        if isinstance(node, dsl_nodes.BoolFeature):
            return node.name

        if isinstance(node, dsl_nodes.Program):
            m = self.parse_node_to_str(node.children[0])
            return f"DEF run m( {m} m)"

        if isinstance(node, dsl_nodes.While):
            c = self.parse_node_to_str(node.children[0])
            w = self.parse_node_to_str(node.children[1])
            return f"WHILE c( {c} c) w( {w} w)"
        if isinstance(node, dsl_nodes.Repeat):
            n = self.parse_node_to_str(node.children[0])
            r = self.parse_node_to_str(node.children[1])
            return f"REPEAT {n} r( {r} r)"
        if isinstance(node, dsl_nodes.If):
            c = self.parse_node_to_str(node.children[0])
            i = self.parse_node_to_str(node.children[1])
            return f"IF c( {c} c) i( {i} i)"
        if isinstance(node, dsl_nodes.ITE):
            c = self.parse_node_to_str(node.children[0])
            i = self.parse_node_to_str(node.children[1])
            e = self.parse_node_to_str(node.children[2])
            return f"IFELSE c( {c} c) i( {i} i) ELSE e( {e} e)"
        if isinstance(node, dsl_nodes.Concatenate):
            s1 = self.parse_node_to_str(node.children[0])
            s2 = self.parse_node_to_str(node.children[1])
            return f"{s1} {s2}"

        if isinstance(node, dsl_nodes.Not):
            c = self.parse_node_to_str(node.children[0])
            return f"not c( {c} c)"
        if isinstance(node, dsl_nodes.And):
            c1 = self.parse_node_to_str(node.children[0])
            c2 = self.parse_node_to_str(node.children[1])
            return f"and c( {c1} c) c( {c2} c)"
        if isinstance(node, dsl_nodes.Or):
            c1 = self.parse_node_to_str(node.children[0])
            c2 = self.parse_node_to_str(node.children[1])
            return f"or c( {c1} c) c( {c2} c)"

        raise Exception(f"Unknown node type: {type(node)}")

    def parse_str_list_to_node(self, prog_str_list: list[str]) -> dsl_nodes.BaseNode:

        if prog_str_list[0] in self.actions:
            if len(prog_str_list) > 1:
                s1 = dsl_nodes.Action(prog_str_list[0])
                s2 = self.parse_str_list_to_node(prog_str_list[1:])
                return dsl_nodes.Concatenate.new(s1, s2)
            return dsl_nodes.Action(prog_str_list[0])

        if prog_str_list[0] in self.bool_features:
            if len(prog_str_list) > 1:
                assert prog_str_list[1] == "h(", "Invalid program"
                assert prog_str_list[-1] == "h)", "Invalid program"
                s1 = prog_str_list[0]
                assert s1 in [
                    "front_object_type",
                    "front_object_color",
                ], "Invalid program"
                assert len(prog_str_list) == 4, "Invalid program"
                s2 = prog_str_list[2]
                if s1 == "front_object_type":
                    return MinigridObjectFeatureNode.new(s1, s2)
                elif s1 == "front_object_color":
                    return MinigridColorFeatureNode.new(s1, s2)

            return dsl_nodes.BoolFeature(prog_str_list[0])

        if prog_str_list[0] in self.int_features:
            if len(prog_str_list) > 1:
                s1 = dsl_nodes.IntFeature(prog_str_list[0])
                s2 = self.parse_str_list_to_node(prog_str_list[1:])
                return dsl_nodes.Concatenate.new(s1, s2)
            return dsl_nodes.IntFeature(prog_str_list[0])

        if prog_str_list[0] == "<HOLE>":
            if len(prog_str_list) > 1:
                s1 = None
                s2 = self.parse_str_list_to_node(prog_str_list[1:])
                return dsl_nodes.Concatenate.new(s1, s2)
            return None

        if prog_str_list[0] == "DEF":
            assert prog_str_list[1] == "run", "Invalid program"
            assert prog_str_list[2] == "m(", "Invalid program"
            assert prog_str_list[-1] == "m)", "Invalid program"
            m = self.parse_str_list_to_node(prog_str_list[3:-1])
            return dsl_nodes.Program.new(m)

        elif prog_str_list[0] == "IF":
            c_end = _find_close_token(prog_str_list, "c", 1)
            i_end = _find_close_token(prog_str_list, "i", c_end + 1)
            c = self.parse_str_list_to_node(prog_str_list[2:c_end])
            i = self.parse_str_list_to_node(prog_str_list[c_end + 2 : i_end])
            if i_end == len(prog_str_list) - 1:
                return dsl_nodes.If.new(c, i)
            else:
                return dsl_nodes.Concatenate.new(
                    dsl_nodes.If.new(c, i),
                    self.parse_str_list_to_node(prog_str_list[i_end + 1 :]),
                )
        elif prog_str_list[0] == "IFELSE":
            c_end = _find_close_token(prog_str_list, "c", 1)
            i_end = _find_close_token(prog_str_list, "i", c_end + 1)
            assert prog_str_list[i_end + 1] == "ELSE", "Invalid program"
            e_end = _find_close_token(prog_str_list, "e", i_end + 2)
            c = self.parse_str_list_to_node(prog_str_list[2:c_end])
            i = self.parse_str_list_to_node(prog_str_list[c_end + 2 : i_end])
            e = self.parse_str_list_to_node(prog_str_list[i_end + 3 : e_end])
            if e_end == len(prog_str_list) - 1:
                return dsl_nodes.ITE.new(c, i, e)
            else:
                return dsl_nodes.Concatenate.new(
                    dsl_nodes.ITE.new(c, i, e),
                    self.parse_str_list_to_node(prog_str_list[e_end + 1 :]),
                )
        elif prog_str_list[0] == "WHILE":
            c_end = _find_close_token(prog_str_list, "c", 1)
            w_end = _find_close_token(prog_str_list, "w", c_end + 1)
            c = self.parse_str_list_to_node(prog_str_list[2:c_end])
            w = self.parse_str_list_to_node(prog_str_list[c_end + 2 : w_end])
            if w_end == len(prog_str_list) - 1:
                return dsl_nodes.While.new(c, w)
            else:
                return dsl_nodes.Concatenate.new(
                    dsl_nodes.While.new(c, w),
                    self.parse_str_list_to_node(prog_str_list[w_end + 1 :]),
                )
        elif prog_str_list[0] == "REPEAT":
            n = self.parse_str_list_to_node([prog_str_list[1]])
            r_end = _find_close_token(prog_str_list, "r", 2)
            r = self.parse_str_list_to_node(prog_str_list[3:r_end])
            if r_end == len(prog_str_list) - 1:
                return dsl_nodes.Repeat.new(n, r)
            else:
                return dsl_nodes.Concatenate.new(
                    dsl_nodes.Repeat.new(n, r),
                    self.parse_str_list_to_node(prog_str_list[r_end + 1 :]),
                )

        elif prog_str_list[0] == "not":
            assert prog_str_list[1] == "c(", "Invalid program"
            assert prog_str_list[-1] == "c)", "Invalid program"
            c = self.parse_str_list_to_node(prog_str_list[2:-1])
            return dsl_nodes.Not.new(c)
        elif prog_str_list[0] == "and":
            c1_end = _find_close_token(prog_str_list, "c", 1)
            assert prog_str_list[c1_end + 1] == "c(", "Invalid program"
            assert prog_str_list[-1] == "c)", "Invalid program"
            c1 = self.parse_str_list_to_node(prog_str_list[2:c1_end])
            c2 = self.parse_str_list_to_node(prog_str_list[c1_end + 2 : -1])
            return dsl_nodes.And.new(c1, c2)
        elif prog_str_list[0] == "or":
            c1_end = _find_close_token(prog_str_list, "c", 1)
            assert prog_str_list[c1_end + 1] == "c(", "Invalid program"
            assert prog_str_list[-1] == "c)", "Invalid program"
            c1 = self.parse_str_list_to_node(prog_str_list[2:c1_end])
            c2 = self.parse_str_list_to_node(prog_str_list[c1_end + 2 : -1])
            return dsl_nodes.Or.new(c1, c2)

        elif prog_str_list[0].startswith("R="):
            num = int(prog_str_list[0].replace("R=", ""))
            assert num is not None
            return dsl_nodes.ConstInt(num)
        elif prog_str_list[0] in ["True", "False"]:
            return dsl_nodes.ConstBool(prog_str_list[0] == "True")
        else:
            raise Exception(f"Unrecognized token: {prog_str_list[0]}.")
