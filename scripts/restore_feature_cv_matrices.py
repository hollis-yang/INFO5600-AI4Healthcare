"""Rebuild purged CV matrix caches from retained raw features and fold preprocessors."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from sepsis_warning.feature_cache import restore_cv_matrices

if __name__ == "__main__":
    for path in restore_cv_matrices():print(path,flush=True)
