import yaml

class Config:
    def __init__(self, config_path: str):
        """
        Initialize the Config class with a path to the YAML configuration file.

        :param config_path: Path to the YAML configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """
        Load the YAML configuration file.

        :return: Configuration dictionary.
        """
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def __getattr__(self, item):
        """
        Allow access to configuration keys as attributes, supporting nested structures.

        :param item: Attribute name to access.
        :return: Value from the configuration or raise AttributeError if not found.
        """
        try:
            value = self.config[item]
            if isinstance(value, dict):
                return ConfigDict(value)
            return value
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{item}'")

    def __str__(self):
        return str(self.config)


class ConfigDict:
    def __init__(self, config_dict):
        self._config_dict = config_dict

    def __getattr__(self, item):
        try:
            value = self._config_dict[item]
            if isinstance(value, dict):
                return ConfigDict(value)
            return value
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{item}'")

    def __str__(self):
        return str(self._config_dict)
