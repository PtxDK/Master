from heartnet.config import YamlConfig
from heartnet.loader.base_loader import load_datasets
import sys

yaml_files = sys.argv[1:]
configs = [YamlConfig.from_file(file) for file in yaml_files]
load_datasets(configs[0])

print(configs[0]["data"])