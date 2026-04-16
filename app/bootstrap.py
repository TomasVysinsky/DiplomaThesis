from pathlib import Path
import sys


def bootstrap_project_paths():
    """
    Ensure both project root and DetekciaOdpaduMiniProjekt folder
    are importable in the same way as in notebooks.
    """
    app_dir = Path(__file__).resolve().parent
    root_dir = app_dir.parent
    project_dir = root_dir / "DetekciaOdpaduMiniProjekt"

    root_str = str(root_dir)
    project_str = str(project_dir)

    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    if project_str not in sys.path:
        sys.path.insert(0, project_str)