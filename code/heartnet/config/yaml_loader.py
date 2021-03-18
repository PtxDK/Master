import yaml

class Loader(yaml.SafeLoader):
    def __init__(self, stream) -> None:
        super().__init__(stream)
    
    def load_extend(self, node):
        print("val")
        print(node)
        
Loader.add_constructor("!extend", Loader.load_extend)