from pathlib import Path
from typing import List, Tuple

import ffmpeg


class FragmentExtractor:
    """Extract and save video fragments using FFmpeg."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def extract_fragment(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        query: str
    ) -> Path:
        """Extract single video fragment and save to query-specific directory."""
        query_dir = self.output_dir / query
        query_dir.mkdir(parents=True, exist_ok=True)
        video_name = video_path.stem
        start_formatted = f"{int(start_time):05d}"
        end_formatted = f"{int(end_time):05d}"
        output_filename = f"{video_name}_{start_formatted}_{end_formatted}.mp4"
        output_path = query_dir / output_filename
        try:
            (
                ffmpeg
                .input(str(video_path), ss=start_time, t=end_time - start_time)
                .output(str(output_path), codec='copy', loglevel='error')
                .overwrite_output()
                .run()
            )
        except ffmpeg.Error as e:
            (
                ffmpeg
                .input(str(video_path), ss=start_time, t=end_time - start_time)
                .output(str(output_path), loglevel='error')
                .overwrite_output()
                .run()
            )
        return output_path


    def extract_batch(
        self,
        results: List[Tuple[Path, float, float, float]],
        query: str
    ) -> List[Path]:
        """Extract multiple video fragments in batch for a query."""
        extracted_paths = []
        for video_path, start, end, score in results:
            output_path = self.extract_fragment(video_path, start, end, query)
            extracted_paths.append(output_path)
        return extracted_paths

