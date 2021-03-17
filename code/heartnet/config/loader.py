import yaml

class YamlConfig:
    def __init__(self, config) -> None:
        self.config = config
        
    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "r") as file:
            config = yaml.full_load(file)
        return cls(config)
        
    def __getitem__(self, idx):
        return self.config[idx]