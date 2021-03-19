import yaml


class Loader(yaml.Loader):

    def __init__(self, stream) -> None:
        super().__init__(stream)

    def load_extend(self, node, extra):
        with open(node, "r") as file:
            base = yaml.load(file, Loader=Loader)
        extension = self.construct_mapping(extra, True)
        return self.merge_dicts(base, extension)

    def merge_dicts(self, dict1, dict2):
        for key1 in dict1:
            if key1 in dict2:
                if isinstance(dict2[key1], dict):
                    dict1[key1] = self.merge_dicts(dict1[key1], dict2[key1])
                else:
                    dict1[key1] = dict2[key1]
        for key2 in dict2:
            if key2 not in dict1:
                dict1[key2] = dict2[key2]
        return dict1

    def construct_python_object(self, suffix, node):
        cls = self.find_python_name(suffix, node.start_mark)
        state = self.construct_mapping(node, deep=True)
        return cls(**state)


Loader.add_multi_constructor("tag:yaml.org,2002:extend:", Loader.load_extend)
Loader.add_multi_constructor(
    'tag:yaml.org,2002:python/object:', Loader.construct_python_object
)
