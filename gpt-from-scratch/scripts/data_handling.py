def get_text(name: str) -> str:
    """
    Read a text file and return its content as a string.
    """
    f_paths = {
        "shakespeare": "./data/shakespeare.txt",
        "jekyll": "./data/jekyll.txt",
        "gatsby": "./data/gatsby.txt",
        "dorian": "./data/dorian-gray.txt",
        "earth": "./data/theory-of-earth.txt",
    }
    f_path = f_paths.get(name, None)

    # Check if the file exists
    if f_path is None:
        raise ValueError(f"Unknown dataset name: {name}. Available options are: {list(f_paths.keys())}")
    
    # Read the file
    with open(f_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text