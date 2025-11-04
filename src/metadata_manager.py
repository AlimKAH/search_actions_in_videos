from pathlib import Path
from typing import List, Tuple

import pandas as pd


class MetadataManager:
    """Manage CSV metadata for extracted video fragments."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.csv_path = output_dir / "metadata.csv"


    def create_metadata(
        self,
        results: List[Tuple[Path, float, float, float]],
        query: str,
        output_paths: List[Path]
    ) -> pd.DataFrame:
        """Create DataFrame with metadata for extracted fragments."""
        records = []
        for (video_path, start, end, score), output_path in zip(results, output_paths):
            records.append({
                "file_path": str(output_path),
                "start_time": start,
                "end_time": end,
                "query": query,
                "similarity_score": score,
                "source_film": video_path.name
            })
        return pd.DataFrame(records)


    def save_metadata(self, df: pd.DataFrame) -> None:
        """Save or append metadata to CSV file."""
        if self.csv_path.exists():
            existing_df = pd.read_csv(self.csv_path)
            df = pd.concat([existing_df, df], ignore_index=True)
        df.to_csv(self.csv_path, index=False)


    def append_metadata(
        self,
        results: List[Tuple[Path, float, float, float]],
        query: str,
        output_paths: List[Path]
    ) -> None:
        """Create and append metadata for search results."""
        df = self.create_metadata(results, query, output_paths)
        self.save_metadata(df)

