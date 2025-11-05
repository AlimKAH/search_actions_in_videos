from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scenedetect import ContentDetector, SceneManager, open_video


class VideoProcessor:
    """Video processing with scene detection and frame extraction."""

    def __init__(
        self,
        scene_threshold: float = 27.0,
        skip_intro_seconds: int = 120,
        skip_outro_seconds: int = 300,
        use_motion_filter: bool = False,
        min_motion_threshold: float = 3.0
    ):
        self.scene_threshold = scene_threshold
        self.skip_intro_seconds = skip_intro_seconds
        self.skip_outro_seconds = skip_outro_seconds
        self.use_motion_filter = use_motion_filter
        self.min_motion_threshold = min_motion_threshold


    def detect_scenes(self, video_path: Path) -> List[Tuple[float, float]]:
        """Detect scene boundaries in video using content-based detection."""
        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.scene_threshold))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
        return [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scene_list]


    def extract_frames(self, video_path: Path, fps: float = 1.0) -> Tuple[List[np.ndarray], List[float]]:
        """Extract frames from video at specified frame rate with timestamps."""
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        start_frame = int(self.skip_intro_seconds * video_fps)
        end_frame = int((duration - self.skip_outro_seconds) * video_fps) if duration > self.skip_outro_seconds else total_frames
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_interval = int(video_fps / fps)
        frames = []
        timestamps = []
        frame_count = start_frame
        prev_frame = None
        
        while frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            if (frame_count - start_frame) % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if self.use_motion_filter and prev_frame is not None:
                    if self._has_significant_motion(prev_frame, frame_rgb):
                        frames.append(frame_rgb)
                        timestamps.append(frame_count / video_fps)
                        prev_frame = frame_rgb
                else:
                    frames.append(frame_rgb)
                    timestamps.append(frame_count / video_fps)
                    prev_frame = frame_rgb
            
            frame_count += 1
        
        cap.release()
        return frames, timestamps


    def get_video_duration(self, video_path: Path) -> float:
        """Get total duration of video in seconds."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return duration


    def generate_fragments(self, scenes: List[Tuple[float, float]], fragment_length: int = 5) -> List[Tuple[float, float]]:
        """Generate non-overlapping fragments of fixed length from detected scenes."""
        fragments = []
        for start, end in scenes:
            scene_duration = end - start
            if scene_duration < fragment_length:
                center = (start + end) / 2
                frag_start = max(0, center - fragment_length / 2)
                frag_end = frag_start + fragment_length
                fragments.append((frag_start, frag_end))
            else:
                num_fragments = int(scene_duration / fragment_length)
                for i in range(num_fragments):
                    frag_start = start + i * fragment_length
                    frag_end = frag_start + fragment_length
                    if frag_end <= end:
                        fragments.append((frag_start, frag_end))
        return self._remove_overlaps(fragments)


    def _remove_overlaps(self, fragments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Remove overlapping fragments keeping earliest occurrences."""
        if not fragments:
            return []
        sorted_fragments = sorted(fragments, key=lambda x: x[0])
        non_overlapping = [sorted_fragments[0]]
        for current in sorted_fragments[1:]:
            if current[0] >= non_overlapping[-1][1]:
                non_overlapping.append(current)
        return non_overlapping


    def _has_significant_motion(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Check if frames have significant motion using optical flow."""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        return np.mean(magnitude) > self.min_motion_threshold

