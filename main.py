import pickle
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from schemas.settings import AppConfig
from src.clip_encoder import CLIPEncoder
from src.fragment_extractor import FragmentExtractor
from src.metadata_manager import MetadataManager
from src.search_engine import SearchEngine
from src.video_processor import VideoProcessor


class VideoSearchSystem:
    """CLIP-based video fragment search system with scene detection."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.video_processor = VideoProcessor(
            config.processing.scene_threshold,
            config.processing.skip_intro_seconds,
            config.processing.skip_outro_seconds,
            config.processing.use_motion_filter,
            config.processing.min_motion_threshold
        )
        self.clip_encoder = CLIPEncoder(config.model.name, config.device)
        self.search_engine = SearchEngine(config.similarity_threshold, config.aggregation_method)
        self.fragment_extractor = FragmentExtractor(config.output_dir)
        self.metadata_manager = MetadataManager(config.output_dir)
        config.cache_dir.mkdir(parents=True, exist_ok=True)


    def index_video(self, video_path: Path) -> None:
        """Index single video by extracting frames and computing CLIP embeddings."""
        cache_file = self.config.cache_dir / f"{video_path.stem}.pkl"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)
            self.search_engine.cache_embeddings(
                video_path,
                cache_data["embeddings"],
                cache_data["timestamps"],
                cache_data.get("scenes")
            )
            return
        try:
            scenes = self.video_processor.detect_scenes(video_path)
            frames, timestamps = self.video_processor.extract_frames(video_path)
            if not frames:
                return
            embeddings = self.clip_encoder.encode_frames(frames, self.config.batch_size)
            self.search_engine.cache_embeddings(video_path, embeddings, timestamps, scenes)
            cache_data = {
                "embeddings": embeddings,
                "timestamps": timestamps,
                "scenes": scenes
            }
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)
        except Exception:
            pass


    def index_all(self, films_dir: Path) -> None:
        """Index all videos in directory with progress tracking."""
        video_files = list(films_dir.glob("*.mp4")) + list(films_dir.glob("*.avi")) + list(films_dir.glob("*.mkv"))
        for video_path in tqdm(video_files, desc="Indexing videos"):
            self.index_video(video_path)


    def search_query(self, query: str, films_dir: Path) -> List:
        """Search for video fragments matching text query and extract results."""
        query_embedding = self.clip_encoder.encode_text(query)
        all_results = []
        video_files = list(films_dir.glob("*.mp4")) + list(films_dir.glob("*.avi")) + list(films_dir.glob("*.mkv"))
        for video_path in tqdm(video_files, desc=f"Searching '{query}'"):
            try:
                embeddings, timestamps = self.search_engine.get_cached_embeddings(video_path)
                if embeddings is None:
                    self.index_video(video_path)
                    embeddings, timestamps = self.search_engine.get_cached_embeddings(video_path)
                if embeddings is None:
                    continue
                
                scenes = self.search_engine.get_cached_scenes(video_path)
                if scenes is None:
                    scenes = self.video_processor.detect_scenes(video_path)
                
                if not scenes:
                    duration = self.video_processor.get_video_duration(video_path)
                    if duration < self.config.fragment_length:
                        continue
                    scenes = [(0, duration)]
                
                fragments = self.video_processor.generate_fragments(scenes, self.config.fragment_length)
                if not fragments:
                    continue
                similarities = self.search_engine.compute_similarity(query_embedding, embeddings)
                scores = self.search_engine.aggregate_fragment_scores(similarities, timestamps, fragments)
                results = self.search_engine.rank_fragments(fragments, scores, video_path)
                all_results.extend(results)
            except Exception:
                continue
        all_results = sorted(all_results, key=lambda x: x[3], reverse=True)
        if all_results:
            output_paths = self.fragment_extractor.extract_batch(all_results, query)
            self.metadata_manager.append_metadata(all_results, query, output_paths)
        return all_results


    def batch_search(self, queries: List[str], films_dir: Path) -> None:
        """Execute search for multiple queries sequentially."""
        for query in queries:
            self.search_query(query, films_dir)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def index(cfg: DictConfig) -> None:
    """Index all videos in films directory."""
    config = AppConfig(**cfg)
    system = VideoSearchSystem(config)
    system.index_all(config.films_dir)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def search(cfg: DictConfig) -> None:
    """Search for video fragments matching query."""
    config = AppConfig(**cfg)
    if not config.query:
        raise ValueError("Query parameter is required. Use: query=explosion")
    system = VideoSearchSystem(config)
    system.search_query(config.query, config.films_dir)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def batch_search(cfg: DictConfig) -> None:
    """Batch search for multiple queries."""
    config = AppConfig(**cfg)
    if not config.queries:
        raise ValueError("Queries parameter is required. Use: queries=[explosion,flood,fight]")
    system = VideoSearchSystem(config)
    system.batch_search(config.queries, config.films_dir)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [options]")
        print("Commands: index, search, batch_search")
        sys.exit(1)
    
    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    if command == "index":
        index()
    elif command == "search":
        search()
    elif command == "batch_search":
        batch_search()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

