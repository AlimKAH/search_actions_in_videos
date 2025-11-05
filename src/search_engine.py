from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


class SearchEngine:
    """Similarity-based search engine for video fragments using CLIP embeddings."""

    def __init__(self, similarity_threshold: float = 0.25, aggregation_method: str = "max"):
        self.similarity_threshold = similarity_threshold
        self.aggregation_method = aggregation_method
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        self.metadata_cache: Dict[str, Dict] = {}


    def compute_similarity(self, query_embedding: np.ndarray, frame_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and frame embeddings."""
        return np.dot(frame_embeddings, query_embedding.T).squeeze()


    def aggregate_fragment_scores(
        self,
        similarities: np.ndarray,
        timestamps: List[float],
        fragments: List[Tuple[float, float]]
    ) -> List[float]:
        """Aggregate similarity scores across fragment time windows."""
        fragment_scores = []
        for frag_start, frag_end in fragments:
            mask = [(frag_start <= t <= frag_end) for t in timestamps]
            if any(mask):
                fragment_similarities = similarities[mask]
                if self.aggregation_method == "mean":
                    fragment_scores.append(float(np.mean(fragment_similarities)))
                else:
                    fragment_scores.append(float(np.max(fragment_similarities)))
            else:
                fragment_scores.append(0.0)
        return fragment_scores


    def rank_fragments(
        self,
        fragments: List[Tuple[float, float]],
        scores: List[float],
        video_path: Path
    ) -> List[Tuple[Path, float, float, float]]:
        """Rank fragments by similarity score and filter by threshold."""
        results = []
        for (start, end), score in zip(fragments, scores):
            if score >= self.similarity_threshold:
                results.append((video_path, start, end, score))
        return sorted(results, key=lambda x: x[3], reverse=True)


    def cache_embeddings(
        self,
        video_path: Path,
        embeddings: np.ndarray,
        timestamps: List[float],
        scenes: List[Tuple[float, float]] = None
    ) -> None:
        """Cache computed embeddings, timestamps, and scene boundaries for video."""
        cache_key = str(video_path)
        self.embeddings_cache[cache_key] = embeddings
        self.metadata_cache[cache_key] = {
            "timestamps": timestamps,
            "scenes": scenes
        }


    def get_cached_embeddings(self, video_path: Path) -> Tuple[np.ndarray, List[float]]:
        """Retrieve cached embeddings and timestamps for video."""
        cache_key = str(video_path)
        if cache_key in self.embeddings_cache:
            embeddings = self.embeddings_cache[cache_key]
            timestamps = self.metadata_cache[cache_key]["timestamps"]
            return embeddings, timestamps
        return None, None


    def get_cached_scenes(self, video_path: Path) -> List[Tuple[float, float]]:
        """Retrieve cached scene boundaries for video."""
        cache_key = str(video_path)
        if cache_key in self.metadata_cache:
            return self.metadata_cache[cache_key].get("scenes")
        return None

