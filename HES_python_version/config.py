from imports import *

class ConfigLoader():
    def __init__(self, filename = "data.json"):
        self.filename = filename

    def load_config(self) -> dict:
        with open(self.filename, 'r') as f:
            config = json.load(f)
        return config