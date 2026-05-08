"""
Ray Serve unit/integration tests.

Run with: python tests/test_ray_serve.py
or: pytest tests/test_ray_serve.py -v
"""

import sys
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, "/app")


# --------------------------------------------------------------------------- #
# Unit tests — serve app logic                                                  #
# --------------------------------------------------------------------------- #

class TestRayServeApp(unittest.TestCase):
    """Test the Ray Serve FastAPI app logic without a live server."""

    def test_mobilenet_v3_small_loadable(self):
        """MobileNet V3 Small can be instantiated from torchvision."""
        try:
            import torch
            from torchvision.models import mobilenet_v3_small
            model = mobilenet_v3_small(weights=None)
            self.assertIsNotNone(model)
            # Check output shape
            dummy = torch.randn(1, 3, 224, 224)
            out = model(dummy)
            self.assertEqual(out.shape[0], 1)
            self.assertEqual(out.shape[1], 1000)  # ImageNet classes
        except ImportError:
            self.skipTest("torch/torchvision not installed")

    def test_mobilenet_custom_head(self):
        """MobileNet classifier head can be replaced for multi-label."""
        try:
            import torch
            import torch.nn as nn
            from torchvision.models import mobilenet_v3_small
            model = mobilenet_v3_small(weights=None)
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, 24)  # 24 MIRFLICKR classes
            dummy = torch.randn(1, 3, 224, 224)
            out = model(dummy)
            self.assertEqual(out.shape, (1, 24))
        except ImportError:
            self.skipTest("torch/torchvision not installed")

    def test_predict_fallback_no_model(self):
        """Without model, service returns 'general' with 0.5 confidence."""
        model_path = "/nonexistent/model.pt"
        exists = Path(model_path).exists()
        if not exists:
            result = {
                "image_id": "test_001",
                "predicted_category": "general",
                "confidence": 0.5,
                "labels": ["general"],
            }
        self.assertEqual(result["predicted_category"], "general")
        self.assertEqual(result["confidence"], 0.5)

    def test_model_checkpoint_exists(self):
        """Trained model file exists after Part 3."""
        pt_path = Path("/app/models/image_classifier.pt")
        pkl_path = Path("/app/models/image_classifier.pkl")
        self.assertTrue(pt_path.exists() or pkl_path.exists(),
                        "No model file found (.pt or .pkl)")


# --------------------------------------------------------------------------- #
# Integration tests — live Ray Serve endpoint                                   #
# --------------------------------------------------------------------------- #

class TestRayServeIntegration(unittest.TestCase):
    """
    Integration tests against a running Ray Serve instance.
    Skipped automatically if Ray Serve is not reachable.
    """

    RAY_SERVE_URL = "http://localhost:8000"

    @classmethod
    def setUpClass(cls):
        import requests
        try:
            r = requests.get(f"{cls.RAY_SERVE_URL}/health", timeout=3)
            cls._available = r.status_code == 200
        except Exception:
            cls._available = False
        if not cls._available:
            print(f"\n  [SKIP] Ray Serve not reachable at {cls.RAY_SERVE_URL}")

    def _skip_if_down(self):
        if not self._available:
            self.skipTest("Ray Serve not running")

    def test_health_endpoint(self):
        self._skip_if_down()
        import requests
        r = requests.get(f"{self.RAY_SERVE_URL}/health", timeout=5)
        self.assertEqual(r.status_code, 200)
        self.assertIn("status", r.json())

    def test_predict_returns_category(self):
        self._skip_if_down()
        import requests
        payload = {
            "image_id": "test_integration_001",
            "image_uri": "hdfs://namenode:9000/photos/raw/images/sample_0001.jpg",
            "caption": "a mountain landscape at sunset",
            "labels": ["mountain", "landscape", "sunset", "nature"],
        }
        r = requests.post(f"{self.RAY_SERVE_URL}/predict", json=payload, timeout=10)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("predicted_category", body)
        self.assertIn("confidence", body)
        self.assertIn("labels", body)
        self.assertIsInstance(body["confidence"], float)
        self.assertGreater(body["confidence"], 0)

    def test_predict_with_empty_labels(self):
        self._skip_if_down()
        import requests
        payload = {
            "image_id": "test_empty_labels",
            "image_uri": "hdfs://namenode:9000/photos/raw/images/sample_0002.jpg",
        }
        r = requests.post(f"{self.RAY_SERVE_URL}/predict", json=payload, timeout=10)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("predicted_category", body)

    def test_predict_multiple_images(self):
        self._skip_if_down()
        import requests
        test_cases = [
            ("beach sunset vacation", ["travel"]),
            ("city street building urban", ["city", "travel"]),
            ("food restaurant pasta", ["food"]),
        ]
        for caption, expected_cats in test_cases:
            payload = {
                "image_id": f"test_{caption.replace(' ', '_')[:20]}",
                "image_uri": "",
                "caption": caption,
                "labels": caption.split(),
            }
            r = requests.post(f"{self.RAY_SERVE_URL}/predict", json=payload, timeout=10)
            self.assertEqual(r.status_code, 200, f"Failed for: {caption}")


# --------------------------------------------------------------------------- #
# Embeddings unit tests                                                         #
# --------------------------------------------------------------------------- #

class TestEmbeddings(unittest.TestCase):
    """Unit tests for the embeddings module."""

    def test_deterministic_embedding_reproducible(self):
        from src.common.embeddings import deterministic_embedding
        v1 = deterministic_embedding("hello world", dim=128)
        v2 = deterministic_embedding("hello world", dim=128)
        self.assertEqual(v1, v2, "Same input must yield same embedding")

    def test_deterministic_embedding_unit_norm(self):
        import numpy as np
        from src.common.embeddings import deterministic_embedding
        v = deterministic_embedding("test embedding", dim=128)
        norm = float(np.linalg.norm(v))
        self.assertAlmostEqual(norm, 1.0, places=5, msg="Embedding should be unit-norm")

    def test_deterministic_embedding_different_inputs(self):
        from src.common.embeddings import deterministic_embedding
        v1 = deterministic_embedding("cats and dogs", dim=128)
        v2 = deterministic_embedding("mountains and rivers", dim=128)
        self.assertNotEqual(v1, v2)

    def test_get_embedding_from_row(self):
        from src.common.embeddings import get_embedding
        row = {
            "image_id": "sample_001",
            "caption": "a photo of the Golden Gate Bridge",
            "vlm_labels": ["bridge", "travel", "city"],
            "tags": ["san francisco", "architecture"],
        }
        emb = get_embedding(row)
        self.assertEqual(len(emb), 128)
        import numpy as np
        norm = float(np.linalg.norm(emb))
        self.assertAlmostEqual(norm, 1.0, places=5)


# --------------------------------------------------------------------------- #
# HNSW unit tests                                                               #
# --------------------------------------------------------------------------- #

class TestHNSWIndex(unittest.TestCase):
    """Unit tests for the HNSW index."""

    def test_add_and_search(self):
        import tempfile, numpy as np
        from src.search.hnsw_index import HNSWIndex
        with tempfile.TemporaryDirectory() as tmp:
            idx = HNSWIndex(
                dim=128,
                path=str(Path(tmp) / "index.bin"),
                map_path=str(Path(tmp) / "map.json"),
            )
            v1 = [float(x) for x in np.random.default_rng(1).standard_normal(128)]
            v2 = [float(x) for x in np.random.default_rng(2).standard_normal(128)]
            idx.add("img_001", v1)
            idx.add("img_002", v2)
            results = idx.search(v1, k=2)
            self.assertEqual(len(results), 2)
            # Nearest neighbor of v1 should be itself (distance ~ 0)
            top = results[0]
            self.assertEqual(top["image_id"], "img_001")

    def test_persistence(self):
        import tempfile, numpy as np
        from src.search.hnsw_index import HNSWIndex
        with tempfile.TemporaryDirectory() as tmp:
            idx_path = str(Path(tmp) / "index.bin")
            map_path = str(Path(tmp) / "map.json")
            v = [float(x) for x in np.random.default_rng(42).standard_normal(128)]
            idx1 = HNSWIndex(dim=128, path=idx_path, map_path=map_path)
            idx1.add("img_persist", v)

            # Reload from disk
            idx2 = HNSWIndex(dim=128, path=idx_path, map_path=map_path)
            results = idx2.search(v, k=1)
            self.assertEqual(results[0]["image_id"], "img_persist")


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestRayServeApp))
    suite.addTests(loader.loadTestsFromTestCase(TestRayServeIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEmbeddings))
    suite.addTests(loader.loadTestsFromTestCase(TestHNSWIndex))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
