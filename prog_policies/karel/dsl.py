from prog_policies.base import BaseDSL, dsl_nodes

class KarelDSL(BaseDSL):
    
    def __init__(self):
        nodes_list = [
            dsl_nodes.While(), dsl_nodes.If(), dsl_nodes.ITE(), dsl_nodes.Repeat(),
            dsl_nodes.Concatenate(), dsl_nodes.Not(), dsl_nodes.Action('move'),
            dsl_nodes.Action('turnLeft'), dsl_nodes.Action('turnRight'),
            dsl_nodes.Action('pickMarker'), dsl_nodes.Action('putMarker'),
            dsl_nodes.BoolFeature('frontIsClear'), dsl_nodes.BoolFeature('leftIsClear'),
            dsl_nodes.BoolFeature('rightIsClear'), dsl_nodes.BoolFeature('markersPresent'),
            dsl_nodes.BoolFeature('noMarkersPresent')
        ] + [dsl_nodes.ConstInt(i) for i in range(20)]
        super().__init__(nodes_list)

    def get_dsl_nodes_probs(self, node_type):
        if node_type == dsl_nodes.StatementNode:
            return {
                dsl_nodes.While: 0.15,
                dsl_nodes.Repeat: 0.03,
                dsl_nodes.Concatenate: 0.5,
                dsl_nodes.If: 0.08,
                dsl_nodes.ITE: 0.04,
                dsl_nodes.Action: 0.2
            }
        elif node_type == dsl_nodes.BoolNode:
            return {
                dsl_nodes.BoolFeature: 0.9,
                dsl_nodes.Not: 0.1
            }
        elif node_type == dsl_nodes.IntNode:
            return {
                dsl_nodes.ConstInt: 1.0
            }

    @property
    def action_probs(self):
        return {
            'move': 0.5,
            'turnLeft': 0.15,
            'turnRight': 0.15,
            'putMarker': 0.1,
            'pickMarker': 0.1
        }
    
    @property
    def bool_feat_probs(self):
        return {
            'frontIsClear': 0.5,
            'leftIsClear': 0.15,
            'rightIsClear': 0.15,
            'markersPresent': 0.1,
            'noMarkersPresent': 0.1
        }
    
    @property
    def const_int_probs(self):
        return {
            i: 1 / 20 for i in range(20)
        }
