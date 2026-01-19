import yaml

def load_config(config_path: str) -> dict[str, dict[str,int]]:
    with open(config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return config