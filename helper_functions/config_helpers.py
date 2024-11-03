import yaml
from typing import LiteralString, Dict, Any


def read_yaml(path: LiteralString | str | bytes) -> Dict[str, Any]:
    """
    Reads a YAML file and stores its content in a dictionary.

    Parameters:
        path (LiteralString | str | bytes): Path of YAML file to be read in.

    Returns:
        out (dict): Resulting dictionary.
    """
    with open(path) as stream:
        try:
            out = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(exc)
    return out
