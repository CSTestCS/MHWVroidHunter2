"""Remember the window's choices between runs (in %LOCALAPPDATA%\\VRoidHunter\\settings.json)."""

import json
import os

from .oodle import cache_dir

FILE_NAME = "settings.json"


def settings_path(folder=None):
    return os.path.join(folder or cache_dir(), FILE_NAME)


def load(folder=None):
    """The saved settings, or {} if there are none or the file is unreadable."""
    try:
        with open(settings_path(folder), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save(values, folder=None):
    """Write the settings; failing to save is never an error worth stopping for."""
    path = settings_path(folder)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(values, f, indent=2)
        os.replace(tmp, path)
        return True
    except OSError:
        return False
