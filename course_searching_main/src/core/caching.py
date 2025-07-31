import joblib
import hashlib
import os
from pathlib import Path
import pandas as pd

class CacheManager:
    def __init__(self, cache_root="cache"):
        self.cache_root = Path(cache_root)
        self.embeddings_dir = self.cache_root / "embeddings"
        self.graphs_dir = self.cache_root / "graphs"
        self._create_dirs()
        
    def _create_dirs(self):
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
    
    def get_embedding_cache_path(self, data_path):
        file_hash = self._file_hash(data_path)
        return self.embeddings_dir / f"{file_hash}.joblib"
    
    def get_graph_cache_path(self, data_path):
        file_hash = self._file_hash(data_path)
        return self.graphs_dir / f"{file_hash}.graphml"
    
    def _file_hash(self, file_path):
        return hashlib.md5(Path(file_path).read_bytes()).hexdigest()
    
    def cache_exists(self, cache_path):
        return cache_path.exists()
    
    def save_cache(self, obj, cache_path):
        joblib.dump(obj, cache_path)
        
    def load_cache(self, cache_path):
        return joblib.load(cache_path)
    
    def should_rebuild(self, source_path, cache_path):
        if not cache_path.exists():
            return True
        source_mtime = os.path.getmtime(source_path)
        cache_mtime = os.path.getmtime(cache_path)
        return source_mtime > cache_mtime