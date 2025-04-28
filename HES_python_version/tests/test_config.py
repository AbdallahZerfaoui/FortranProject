from config import ConfigLoader
import pytest
import json

def test_config_loader(tmp_path):
	""" 
 	Test the ConfigLoader class to ensure it loads the configuration file correctly.
	"""
	tmp_file = tmp_path/"test_config.json"
	loader = ConfigLoader(filename=tmp_file)
	config = loader.load_config()
 
	expected_config = {
        "Nx": 10,
        "Ny": 15,
        "Lx": 1.0,
        "Ly": 2.0,
        "D": 1.5
    }
    
	assert isinstance(config, dict)
	assert 'Nx' in config
	assert 'Ny' in config
	assert 'Lx' in config
	assert 'Ly' in config
	assert 'D' in config
	assert len(config) == len(expected_config)

	assert config == expected_config
 



