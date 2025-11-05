from typing import List, Union

import clip
import numpy as np
import torch
from PIL import Image


class CLIPEncoder:
    """CLIP model wrapper for encoding frames and text queries."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()


    def encode_frames(self, frames: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Encode video frames into CLIP embeddings with GPU batch processing."""
        embeddings = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            images = [self.preprocess(Image.fromarray(frame)) for frame in batch]
            image_tensor = torch.stack(images).to(self.device)
            with torch.no_grad():
                batch_embeddings = self.model.encode_image(image_tensor)
                batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=-1, keepdim=True)
            embeddings.append(batch_embeddings.cpu().numpy())
        return np.vstack(embeddings)


    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Encode text queries into CLIP embeddings."""
        if isinstance(text, str):
            text = [text]
        text_tokens = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_tokens)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings.cpu().numpy()

