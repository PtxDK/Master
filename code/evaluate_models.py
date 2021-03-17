from heartnet.config import YamlConfig
import sys

yaml_files = sys.argv[1:]
configs = [YamlConfig.from_file(file) for file in yaml_files]
print(configs)