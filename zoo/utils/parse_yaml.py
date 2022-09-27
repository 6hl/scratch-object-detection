import yaml

def parse_yaml(yaml_file_path):
    with open(yaml_file_path) as f:
        return yaml.safe_load(f)