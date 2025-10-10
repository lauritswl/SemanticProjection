import os
import pandas as pd
from src.projecter import ProjectionAnalyzer
class ConceptProjector:
    """
    A class to handle the end-to-end process of loading data, embedding it, projecting concepts, and saving results.
    args:
        CorpusLoader: An instance of the CorpusLoader class to load and preprocess the dataset.
        Embedder: An instance of the Embedder class to generate embeddings for the text data.
        project_name: A string to name the project, used in saving files.
        force_recompute: A boolean flag to force recomputation and overwriting of existing files.
    """
    def __init__(self, CorpusLoader, Embedder, project_name = "Fiction4Sentiment", category_col = None, force_recompute=False, use_saved_concept_vector=False, concept_vector_path=None):
        self.CorpusLoader = CorpusLoader
        self.Embedder = Embedder
        self.ProjectionAnalyzer = ProjectionAnalyzer
        self.project_name = project_name
        self.force_recompute = force_recompute
        self.results = None
        self.category_col = category_col
        self.use_saved_concept_vector = use_saved_concept_vector
        if concept_vector_path is not None:
            self.concept_vector_path = concept_vector_path
        else:
            self.concept_vector_path = f"data/concept_vectors/vectors/{self.project_name}_concept_vector.csv"



    def run(self):        
        # --- Embed the Data ---:
        # --- Embed the training set (binary labels)
        if self.CorpusLoader.train_texts is not None:
            train_EMBEDDED = self.Embedder.embed(
                texts=self.CorpusLoader.train_texts,  
                cache_path=f"data/embeddings/training_{self.project_name}.csv")
            train_EMBEDDED["label"] = self.CorpusLoader.train_labels

        # --- Test set (continuous labels)
        test_EMBEDDED = self.Embedder.embed(
            texts=self.CorpusLoader.test_texts, 
            cache_path=f"data/embeddings/test_{self.project_name}.csv")
        test_EMBEDDED["label"] = self.CorpusLoader.test_labels


        # --- Projection Analysis ---
        if self.use_saved_concept_vector and os.path.exists(self.concept_vector_path):
            self.ProjectionAnalyzer = ProjectionAnalyzer(matrix_project=test_EMBEDDED, use_concept_vector=True, concept_vector_path=self.concept_vector_path)
        else:
            self.ProjectionAnalyzer = ProjectionAnalyzer(matrix_concept=train_EMBEDDED,matrix_project=test_EMBEDDED)        
        self.ProjectionAnalyzer.project()

        # --- Save Results for Reproducibility ---
        if not self.use_saved_concept_vector:
            concept_text = pd.DataFrame({
                "text": self.CorpusLoader.train_texts,
                "label": self.CorpusLoader.train_labels
            })
            self._save_if_needed(
                concept_text,
                f"data/concept_vectors/text/{self.project_name}_concept_text.csv",
                "Concept texts"
            )

            self._save_if_needed(
                self.ProjectionAnalyzer.concept_vector,
                f"data/concept_vectors/vectors/{self.project_name}_concept_vector.csv",
                "Concept vectors"
            )

        results_dict = {
        "text": self.CorpusLoader.test_texts,
        "label": self.CorpusLoader.test_labels,
        "prediction": self.ProjectionAnalyzer.projected_in_1D
        }
        if self.category_col is not None and self.category_col in self.CorpusLoader.test_df.columns:
            results_dict["category"] = self.CorpusLoader.test_df[self.category_col]
        self.results = pd.DataFrame(results_dict)

        self._save_if_needed(
            self.results,
            f"data/processed/{self.project_name}.csv",
            "Results"
        )

        return self.results


    def _save_if_needed(self, df, path, name):
        """Helper: save DataFrame if file doesn't exist or overwrite if forced."""
        if not os.path.exists(path) or self.force_recompute:
            df.to_csv(path, index=False)
            print(f"{name} saved successfully at {path}.")
        else:
            print(f"{name} already exists at {path}. Use force_recompute=True to overwrite.")
