from pathlib import Path
        
def get_base_path():
    """
    Finds the project root directory relative to this utils.py
    file.
    """
    return Path(__file__).resolve().parents[2]

def resolve_path(base_path, p):
    """Resolves relative paths to the project root, keeping absolute paths intact."""
    if p is None:
        return None
        
    path_obj = Path(p)
    if path_obj.is_absolute():
        return path_obj.resolve()
    
    repo_relative = base_path / path_obj
    if repo_relative.exists():
        return repo_relative.resolve()
    
    parent_relative = base_path.parent / path_obj
    if parent_relative.exists():
        return parent_relative.resolve()
        
    return repo_relative