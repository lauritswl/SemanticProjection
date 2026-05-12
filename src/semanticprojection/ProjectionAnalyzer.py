import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
class ProjectionAnalyzer:
    """
    Class to handle projection of matrices onto a concept vector defined by a concept matrix or concept vector.
    
    Using a Concept Matrix:
        A dataframe containing the embeddings of sentences representing a specific concept (e.g., sentiment).        
        The Concept Matrix dataframe should contain examples of two classes (e.g., "positive" and "negative").
        The shape should be [num_examples, embedding_dimensions + 1], where the last column contains the labels.
        Using the This saves a LLM-model specific concept vector in /data/concept_vectors/ folder for later use. 
    Using a Concept Vector:
        The concept vector can be directly provided to project other matrices onto this vector.
        This allows for off-the-shelf projection without needing the original concept examples.
    
    arguments:
        matrix_concept: pandas.DataFrame
            DataFrame containing the embeddings of the concept examples with a labels as the last column (df[-1]==label].
        matrix_project: pandas.DataFrame
            DataFrame containing the embeddings of the sentences to be projected(/analysed).
    attributes:
        concept_df: pandas.DataFrame
            DataFrame containing the concept examples (embeddings + label).
        label_col: str
            Name of the label column in the concept DataFrame.
        concept_vector: pandas.DataFrame
            The computed concept vector from the concept DataFrame.
        projection_df: pandas.DataFrame
            DataFrame containing the embeddings to be projected.
        projected_matrix: pandas.DataFrame
            The projected matrix onto the concept vector.
        projected_in_1D: pandas.Series
            The 1D coordinates of the projections in the subspace defined by the concept vector.
        projection_distance: pandas.Series
            The distances of the original points to their projections.
    methods:
        set_concept_df(matrix_concept)
            Set the concept DataFrame and initialize label column.
        set_projection_df(matrix_project)
            Set the matrix to project and calculate projections.
        project()
            Project the matrix onto the concept vector and return projections and 1D coordinates.
        define_concept_vector()
            Compute the concept vector from the concept DataFrame.
        define_logistic_concept_vector()
            Compute the concept vector using logistic regression.
    """
    def __init__(self, matrix_concept=None, use_concept_vector = False, concept_vector_path = None, matrix_project=None):
        self.concept_df = matrix_concept
        self.label_col = None
        self.concept_vector = None
        self.projection_df = matrix_project
        self.projected_matrix = None
        self.projected_in_1D = None
        self.projection_distance = None
        self.use_concept_vector = use_concept_vector
        self.concept_vector_path = concept_vector_path
        if matrix_concept is not None:
            self.label_col = matrix_concept.columns[-1]  # Assuming the last column is the label

        if matrix_concept is not None:
            self.concept_vector = self.define_concept_vector()
        if self.use_concept_vector:
            if self.concept_vector_path is not None and os.path.exists(self.concept_vector_path):
                self.concept_vector = pd.read_csv(self.concept_vector_path)
            else:
                raise ValueError("concept_vector_path must be provided when use_concept_vector is True. Find the path in pipeline.concept_vector_path.")


    def set_concept_df(self, matrix_concept):
        """Set the concept dataframe and initialize label column."""
        self.concept_df = matrix_concept
        self.label_col = matrix_concept.columns[-1]  # Assuming the last column is the label
        self.concept_vector = self.define_concept_vector()

    def set_projection_df(self, matrix_project):
        """Set the matrix to project and calculate projections."""
        self.projection_df = matrix_project


    def project(self):
        """
        Project the matrix onto the concept vector.
        Returns both the projection (rank = 1) and the corresponding 1D subspace coordinates of the projections.
        """
        if self.concept_vector is None or self.projection_df is None:
            raise ValueError("Both concept and projection matrices must be set.")
        # Run tests
        length_check_result = self._check_matrices_length()
        if length_check_result is not None:
            raise ValueError(length_check_result)
        if self.use_concept_vector is False:
            label_check_result = self._check_labels_structure()
            if label_check_result is not None:
                raise ValueError(label_check_result)

        # Proceed with the projection if all tests pass
        self.projected_matrix, self.projected_in_1D, self.projection_distance = self._express_matrix_by_vector(self.projection_df.iloc[:, :-1], self.concept_vector)
        return self.projected_matrix, self.projected_in_1D

    def define_concept_vector(self):
        """
        Computes the vector pointing from the mean of the negative class to the mean of the positive class.
        Returns:
        --------
        pandas.DataFrame
            A single-row DataFrame representing the sentiment direction vector.
        """
        # Step 1: Split into positive and negative classes
        pos = self.concept_df[self.concept_df[self.label_col] == "positive"].iloc[:, :-1]
        neg = self.concept_df[self.concept_df[self.label_col] == "negative"].iloc[:, :-1]
        # Step 2: Compute the mean-difference vector
        return self._positive_to_negative_vector(pos, neg)
    
    @staticmethod
    def _positive_to_negative_vector(positive, negative):
        """Compute the mean-difference vector from negative to positive."""
        posneg_vector = positive.mean().to_frame().T - negative.mean().to_frame().T
        return pd.DataFrame(posneg_vector)

    def _express_matrix_by_vector(self, matrix, vector):
        """Return both the projection (rank = 1) and the corresponding 1D subspace coordinates of the projections."""
        unit_vector = vector / np.linalg.norm(vector)  # Step 1: Normalize the vector
        projection, projection_distance = self._project_matrix_to_vector_with_distance(matrix, vector)  # Step 2: Project the matrix onto the vector
        projection_in_1D_subspace = projection.iloc[:, 0] / unit_vector.iloc[:, 0][0]  # Step 3: Compute the projection in the 1D subspace
        return projection, projection_in_1D_subspace, projection_distance  # Step 4: Return both projections

    @staticmethod
    def _project_matrix_to_vector_with_distance(matrix, vector):
        """Project matrix onto subspace spanned by the vector and compute distances."""
        m = matrix.to_numpy()
        v = vector.to_numpy()[0]
        v_dot_v = np.dot(v, v)
        
        # Compute projection
        proj = np.outer(np.dot(m, v) / v_dot_v, v)
        
        # Compute residuals and distances
        residuals = m - proj
        distances = np.linalg.norm(residuals, axis=1)
        
        # Return projection as DataFrame and distances as Series
        return pd.DataFrame(proj, columns=matrix.columns), pd.Series(distances, index=matrix.index)


    def _check_matrices_length(self):
        """Ensure that matrix_concept and matrix_project have the same number of columns."""
        if self.concept_vector.shape[1] != self.projection_df.iloc[:, :-1].shape[1]:
            return f"Both matrices need to be embedded by the same model to ensure similar length of embedding space, concept_vector = {self.concept_vector.shape[1]}, projection_df = {self.projection_df.shape[1]}"
        return None

    def _check_labels_structure(self):
        """Ensure that the last column of both matrices contains 'positive' and 'negative' labels."""
        # Check if the label column exists and contains valid labels
        valid_labels = {"positive", "negative"}
        concept_labels = set(self.concept_df[self.label_col].unique())

        
        if not concept_labels.issubset(valid_labels):
            return "The last column of matrix_concept must contain only 'positive' and 'negative' labels."

        return None
    

    def define_logistic_concept_vector(self):
        # Assume the last column is the label
        X = self.concept_df.iloc[:, :-1].values
        y = (self.concept_df[self.label_col] == "positive").astype(int).values
        clf = LogisticRegression(solver='liblinear')
        clf.fit(X, y)
        # The coefficients are the concept direction
        coef = clf.coef_.flatten()
        # Return as a DataFrame for compatibility
        return pd.DataFrame([coef], columns=self.concept_df.columns[:-1])

