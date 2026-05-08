import hnswlib, numpy as np, json
from pathlib import Path

class HNSWIndex:
    def __init__(self, dim=128, path="outputs/hnsw_index.bin", map_path="outputs/id_map.json"):
        self.dim = dim
        self.path = Path(path)
        self.map_path = Path(map_path)
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.id_map = {}
        if self.path.exists():
            self.index.load_index(str(self.path))
            self.id_map = json.loads(self.map_path.read_text()) if self.map_path.exists() else {}
        else:
            self.index.init_index(max_elements=100000, ef_construction=200, M=16)
            self.index.set_ef(100)

    def add(self, external_id, vector, auto_save=True):
        existing = {v for v in self.id_map.values()}
        if external_id in existing:
            return
        internal_id = len(self.id_map)
        self.id_map[str(internal_id)] = external_id
        self.index.add_items(np.array([vector], dtype=np.float32), np.array([internal_id]))
        if auto_save:
            self.save()

    def search(self, vector, k=10):
        labels, distances = self.index.knn_query(np.array([vector], dtype=np.float32), k=min(k, max(1, len(self.id_map))))
        return [{"image_id": self.id_map.get(str(int(i))), "distance": float(d)} for i, d in zip(labels[0], distances[0])]

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.index.save_index(str(self.path))
        self.map_path.write_text(json.dumps(self.id_map, indent=2))
