import json

from colorama import Fore, Style, init


def log_config(config: dict):
    init(autoreset=True)
    print(f"{Fore.GREEN}{Style.BRIGHT} Config:")
    print(
        f"{Fore.GREEN}{Style.BRIGHT}Hyperparameters:"
        f"{Style.NORMAL}{json.dumps(config, sort_keys=True, indent=4)}{Style.RESET_ALL}"
    )
