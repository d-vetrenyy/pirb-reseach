from pathlib import Path

def get_dotenv(path: str|Path = "./.env"):
    path = Path(path)
    with open(path, 'r') as dotenv_file:
        return {k: v for k, v in [line.split('=') for line in dotenv_file.readlines()]}
