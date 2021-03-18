from heartnet.config import YamlConfig
# from heartnet.loader.base_loader import load_datasets
import sys

yaml_files = sys.argv[1:]
configs = [YamlConfig.from_file(file) for file in yaml_files]
# datasets = load_datasets(configs[0])
# for dataset in datasets.values():
#     for x,y in dataset.take(1):
#         print(x.shape, y.shape)