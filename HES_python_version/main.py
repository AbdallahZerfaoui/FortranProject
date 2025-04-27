from imports import *


def main():
    # Load configuration
	config_loader = ConfigLoader()
	config = config_loader.load_config()
	print(config)
 

if __name__ == "__main__":
    main()