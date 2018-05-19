import os

def get_env(file_path=".env"):
    """ read the content of the .env file,
        and put each env var in a dic

        Args:
            file_path -- string -- the path to the env file

        Return the dic containing the env variables
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError
    env = {}
    with open(file_path, "r") as f:
        for line in f.readlines():
            line = line.strip().split("=")
            # check if the line is a key=value line
            if len(line) == 2:
                # convert from string to int if it's an int
                try:
                    line[1] = int(line[1])
                except ValueError:
                    # convert from string to float if it's a float
                    try:
                        line[1] = float(line[1])
                    except ValueError:
                        pass
                env[line[0]] = line[1]
    return env
