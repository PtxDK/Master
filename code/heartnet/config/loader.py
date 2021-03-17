import yaml

class YamlConfig:
    def __init__(self, config) -> None:
        self.config = config
        print(config)
        
    @staticmethod
    def from_file(cls, file_path):
        with open(file_path, "r") as file:
            config = yaml.load(file)
        return cls(config)
        