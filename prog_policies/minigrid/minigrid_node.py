from prog_policies.base import dsl_nodes


class MinigridObjectFeatureNode(dsl_nodes.BoolFeature, dsl_nodes.TerminalNode):

    def __init__(self, name: str):
        super(dsl_nodes.BoolFeature, self).__init__(name)
        self.object = "object"

    @classmethod
    def new(cls, name: str, object: str):
        node = cls(name)
        node.object = object
        return node

    def interpret(self, env):
        return env.get_bool_feature(self.name)(self.object)


class MinigridColorFeatureNode(dsl_nodes.BoolFeature, dsl_nodes.TerminalNode):

    def __init__(self, name: str):
        super(dsl_nodes.BoolFeature, self).__init__(name)
        self.color = "color"

    @classmethod
    def new(cls, name: str, color: str):
        node = cls(name)
        node.color = color
        return node

    def interpret(self, env):
        return env.get_bool_feature(self.name)(self.color)
