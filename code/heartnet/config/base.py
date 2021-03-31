from typing import List
import yaml
from .yaml_loader import Loader

class YamlConfig:
    def __init__(self, config) -> None:
        self.config = config
        
    @classmethod
    def from_file(cls, file_path):
        with open(file_path, "r") as file:
            config = yaml.load(file, Loader=Loader)
        return cls(config)
        
    def __getitem__(self, idx):
        return self.config[idx]    