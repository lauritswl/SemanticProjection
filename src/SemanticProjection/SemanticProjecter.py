from .Embedder import Embedder
from .ProjectionAnalyzer import ProjectionAnalyzer
import pandas as pd
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
    
    def project_texts(self, texts, concept_vector):
        """
        Embeds and projects a list of texts onto the sentiment concept vector.
        
        Args:
            texts: list of str
                The texts to be embedded and projected.
            concept_vector: str
                The name of semantic concept vector to use for projection.
        Returns:
            pandas.DataFrame
                A DataFrame containing the original texts and their projections.
         """
        embeddings = self.embed_texts(texts)
        projections = self.project_embeddings(embeddings, concept_vector)
        results = pd.DataFrame({
            "text": texts,
            "projection": projections
        })
        return results
    
    def embed_texts(self, texts):
        """
        Embeds a list of texts.
        
        Args:
            texts: list of str
                The texts to be embedded.
        Returns:
            pandas.DataFrame
                A DataFrame containing the embeddings and original texts.
        """
        embeddings = self.embedder.embed(
            texts=texts)
        embeddings["text"] = texts  # Append text to embeddings
        return embeddings

    def project_embeddings(self, embeddings, concept_vector):
        """
        Projects the given embeddings onto the sentiment concept vector and returns the projections.
        
        Args:
            embeddings: pandas.DataFrame
                The DataFrame containing the embeddings to be projected.
            concept_vector: str
                The name of semantic concept vector to use for projection.
        
        Returns:
            pandas.Series
                A Series containing the 1D coordinates of the projections.
        """
        self.projector = ProjectionAnalyzer(
            matrix_project=embeddings,
            use_concept_vector=True,
            concept_vector_path=f"src/SemanticProjection/data/vectors/{concept_vector}.csv"
        )
        self.projector.project()
        return self.projector.projected_in_1D
