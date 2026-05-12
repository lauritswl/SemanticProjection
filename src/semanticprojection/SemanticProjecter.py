from .Embedder import Embedder
from .ProjectionAnalyzer import ProjectionAnalyzer
import pandas as pd
from pathlib import Path
class SemanticProjector:
    """
    Class to handle the end-to-end process of embedding texts and projecting them onto a concept vector.
    
    This class uses the Embedder class to create embeddings for a list of texts and then uses the ProjectionAnalyzer
    class to project these embeddings onto a predefined sentiment concept vector.
    
    Attributes:
        embedder: Embedder
            An instance of the Embedder class for creating text embeddings.
        projector: ProjectionAnalyzer
            An instance of the ProjectionAnalyzer class for projecting embeddings onto a concept vector.
    
    Methods:
        embed_texts(texts, project_name)
            Embeds a list of texts and saves the embeddings to a CSV file.
        project_embeddings(embeddings, concept_vector_path)
            Projects the given embeddings onto the sentiment concept vector and returns the projections.
    """
    def __init__(self, model_name="paraphrase-multilingual-mpnet-base-v2"):
        self.embedder = Embedder(model_name=model_name)
        self.projector = None
        self.results = None
        # Path to the folder containing all concept vector CSVs
        self.vectors_dir = Path(__file__).parent / "data" / "vectors"
        self.force_recompute_flag = True

    def project_texts(self, texts, concept_vector):
        """Embeds and projects a list of texts onto a concept vector."""
        embeddings = self.embed_texts(texts)
        vector_path = self.get_vector_path(concept_vector)
        projections = self.project_embeddings(embeddings, vector_path)
        results = pd.DataFrame({
            "text": texts,
            "projection": projections
        })
        self.results = results
        return results
    
    def standardize(self):
        """Standardizes the projection results to have mean 0 and std 1."""
        if self.results is None:
            raise ValueError("No results to standardize. Please run project_texts first.")
        self.results["standardized_projection"] = (self.results["projection"] - self.results["projection"].mean()) / self.results["projection"].std()
        return self.results
    
    def embed_texts(self, texts):
        """Embeds a list of texts."""
        embeddings = self.embedder.embed(texts=texts)
        embeddings["text"] = texts  # Append text to embeddings
        return embeddings

    def project_embeddings(self, embeddings, concept_vector_path):
        """Projects the embeddings onto the specified concept vector CSV."""
        self.projector = ProjectionAnalyzer(
            matrix_project=embeddings,
            use_concept_vector=True,
            concept_vector_path=str(concept_vector_path)
        )
        self.projector.project()
        return self.projector.projected_in_1D

    def list_vectors(self):
        """List all available concept vectors."""
        return [f.stem for f in self.vectors_dir.glob("*.csv")]
    
    def get_vector_path(self, vector_name):
        """Get full path to a vector CSV from its name."""
        path = self.vectors_dir / f"{vector_name}.csv"
        if not path.exists():
            raise ValueError(f"Vector '{vector_name}' not found. Available vectors: {self.list_vectors()}")
        return path
    
    def set_force_recompute(self, force_recompute):
        self.force_recompute_flag = force_recompute