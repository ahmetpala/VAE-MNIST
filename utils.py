import yaml


def load_config():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)
