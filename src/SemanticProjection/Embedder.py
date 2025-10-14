# %%
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

class Embedder:
    """Class to handle text embedding using SentenceTransformers with caching and hidden state extraction."""
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available
        self.model = SentenceTransformer(model_name, device=self.device) # Load model

    def embed(self, texts, cache_path=None, force_recompute=False):
        """
        Embeds a list of texts into vectors.
        If cache_path is provided, saves/loads a CSV file with embeddings.
        If force_recompute is True, ignores the cache and recomputes embeddings.
        Returns a DataFrame with embeddings.
        """
        if cache_path and os.path.exists(cache_path) and not force_recompute:
            return pd.read_csv(cache_path)

        embeddings = self.model.encode(texts, show_progress_bar=True, device=self.device)
        df = pd.DataFrame(embeddings)

        if cache_path:
            df.to_csv(cache_path, index=False)

        return df