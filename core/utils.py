import os

def ensure_local_dir(path: str):
    """Crea la carpeta local si no existe."""
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
