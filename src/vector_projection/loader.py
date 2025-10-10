import re
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

class CorpusLoader:
    """
    Main functionality:
    Loads, cleans, splits, and formats text-label datasets from CSV or Hugging Face.
    Tehn standardizes columns to ['sentence', 'label'] for embedding and projection, to be used in conjunction with the rest of the codebase.
    Supports both categorical and continuous labels.
    Args ():
    text_col: str, name of the text column in the input data
    label_col: str, name of the label column in the input data
    path: str or None, path to CSV file (default None)
    """
    def __init__(self, path=None, text_col=None, label_col=None):
        self.path = path
        self.text_col = text_col
        self.label_col = label_col
        self.df = None
        self.df_cache = None
        self.hugginface = None
        self.train_df = None
        self.test_df = None
        if self.text_col is None or self.label_col is None:
            raise ValueError("Column names must be specified for 'text' and 'label'.")

    def load_csv(self, start=0, n_rows=None):
        """
        Loads data from a CSV file, optionally starting from a specific row and loading a specific number of rows.
        Standardizes column names to 'sentence' and 'label', recuires input from __self__ defined when creating object.
        Args:
            start: int, starting row index (default 0)
            n_rows: int or None, number of rows to load (default None, loads all)
        Returns:
            pd.DataFrame with standardized columns ['sentence', 'label']
        """
        # Test if file exists
        try:
            df = pd.read_csv(self.path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found at {self.path}")
        if start >= len(df):
            raise IndexError(f"Start index {start} is beyond dataset length {len(df)}.")
        
        # Variable end index (for partial loading if specified)
        end = start + n_rows if n_rows is not None else len(df) 
        df = df.iloc[start:end].reset_index(drop=True)

        # Check if text and label columns exist
        if self.text_col not in df.columns or self.label_col not in df.columns:
            raise ValueError(f"CSV must contain columns '{self.text_col}' and '{self.label_col}'.")
            print(f"Available columns: {df.columns.tolist()}")

        # Standardize column names
        df.rename(columns={self.text_col: 'sentence', self.label_col: 'label'}, inplace=True)
        self.text_col = 'sentence'
        self.label_col = 'label'
        self.df = df
        return self.df

    def load_from_huggingface(self, dataset_name, split="train"):
        """
        Loads dataset from Hugging Face, standardizes column names to 'sentence' and 'label
        Args:
            dataset_name: str, name of the dataset on Hugging Face (e.g., "chcaa/fiction4sentiment")
            split: str, which split to load (default "train")
        Returns:
            pd.DataFrame with standardized columns ['sentence', 'label']
        """

        # Load dataset from Hugging Face
        try:
            self.hugginface = load_dataset(dataset_name, split=split)
        except Exception as e:
            raise RuntimeError(f"Error loading dataset from Hugging Face: {str(e)}")

        # Check if text and label columns exist
        if self.text_col not in self.hugginface.column_names or self.label_col not in self.hugginface.column_names:
            raise ValueError(f"Dataset must contain columns '{self.text_col}' and '{self.label_col}'.")
            print(f"Available columns: {self.hugginface.column_names}")
        
        # Standardize column names
        df = pd.DataFrame(self.hugginface)
        df.rename(columns={self.text_col: 'sentence', self.label_col: 'label'}, inplace=True)
        self.text_col = 'sentence'
        self.label_col = 'label'
        self.df = df
        return df

    def load_from_dataframe(self, df):
        """
        Loads data from an existing pandas DataFrame, standardizes column names to 'sentence' and 'label'
        Args:
            df: pd.DataFrame, input DataFrame
        Returns:
            pd.DataFrame with standardized columns ['sentence', 'label']
        """
        # Check if input is a DataFrame
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        self.df = df.copy()
    
        # Check if text and label columns exist
        if self.text_col not in self.df.columns or self.label_col not in self.df.columns:
            raise ValueError(f"DataFrame must contain columns '{self.text_col}' and '{self.label_col}'.")
            print(f"Available columns: {self.df.columns.tolist()}")
        
        # Standardize column names
        self.df.rename(columns={self.text_col: 'sentence', self.label_col: 'label'}, inplace=True)
        self.text_col = 'sentence'
        self.label_col = 'label'
        return self.df

    def _clean_string(self, raw_string):
        """Cleans a single string by removing HTML tags, slashes, non-alphanumeric characters, and converting to lowercase."""
        if not isinstance(raw_string, str):
            raw_string = str(raw_string) if pd.notnull(raw_string) else ""
        clean_text = re.sub(r'<br\s*/?>', '', raw_string)
        clean_text = re.sub(r'/', ' ', clean_text)
        clean_text = re.sub(r'[^a-zA-Z0-9 ]', '', clean_text).lower()
        return clean_text

    def clean(self):
        """Cleans the text column in the DataFrame by applying _clean_string to each entry."""
        if self.df is None:
            raise ValueError("Data not loaded. Call `load_csv()` first.")
        if self.text_col not in self.df.columns:
            raise ValueError(f"Text column '{self.text_col}' not found in DataFrame.")
        self.df[self.text_col] = self.df[self.text_col].apply(self._clean_string)
        return self.df

    def reduce_to_columns(self):
        """Reduces the DataFrame to only the text and label columns."""
        if self.df is None:
            raise ValueError("Data not loaded.")
        if self.df_cache is None:
            self.df_cache = self.df
        self.df = self.df[[self.text_col, self.label_col]].copy()
        return self.df

    def split_binary_train_continuous_test(self, positive_threshold=0.5, negative_threshold=0.5, train_size=0.6, random_state=42):
        """
        Splits the dataset into a training set with binary labels (positive/negative) and a test set with continuous labels.
        Neutral labels (between negative_threshold and positive_threshold) are excluded from the training set.
        Args:
            positive_threshold: float, threshold above which labels are considered positive
            negative_threshold: float, threshold below which labels are considered negative
            train_size: float, proportion of data to use for training (default 0.6)
            random_state: int, for reproducibility
        Returns:
            train_df, test_df (train_df with columns ['sentence', 'label'] where label is binary, test_df with continuous labels)
        """ 
        if self.df is None:
            raise ValueError("Data not loaded.")

        # Create copy and add binary labels
        df = self.df.copy()
        df['continuous_label'] = df[self.label_col]
        df['binary_label'] = df['continuous_label'].apply(
            lambda x: "positive" if x >= positive_threshold else ("negative" if x <= negative_threshold else "neutral")
        )
        # Create a binary DataFrame without neutral labels, to avoid wasting data during train/test split:
        binary_df = df[df['binary_label'] != 'neutral'].reset_index(drop=True)
        train_df, heldout_df = train_test_split(binary_df, train_size=train_size, stratify=binary_df['binary_label'], random_state=random_state)
        
        # Remove train data from test data:
        test_df = df.drop(index=train_df.index).reset_index(drop=True)
        # Reset index for train_df
        train_df = train_df.reset_index(drop=True)

        # Rename train and test DataFrames to standardize column names in compliance with the rest of the code: 
        self.train_df = train_df.rename(columns={self.text_col: 'sentence'})
        self.train_df['label'] = self.train_df['binary_label']

        self.test_df = test_df.rename(columns={self.text_col: 'sentence'})
        self.test_df['label'] = self.test_df['continuous_label']

        return self.train_df, self.test_df

    def split_extremes_train_middle_test(self, extreme_frac=0.2, remove_frac_from_extremes=0.0, random_state=42):
        """
        Splits the dataset so that the train set contains the lowest and highest extremes (by label),
        each of size (extreme_frac/2), and the test set contains the middle plus optionally a fraction
        of the extremes (remove_frac_from_extremes).
        
        Args:
            extreme_frac: float, total fraction of data to use as extremes (e.g., 0.2 for 10% min + 10% max)
            remove_frac_from_extremes: float, fraction of the extremes to move to the test set (e.g., 0.5 for half)
            random_state: int, for reproducibility
        Returns:
            train_df, test_df (both with columns ['sentence', 'label'])
        """
        if self.df is None:
            raise ValueError("Data not loaded.")
        df = self.df.copy()
        n = len(df)
        n_extreme = int(n * extreme_frac / 2)
        if n_extreme < 1:
            raise ValueError("Not enough samples for the requested extreme fraction.")
        # Sort by label
        df_sorted = df.sort_values(self.label_col, ascending=True)
        min_extremes = df_sorted.iloc[:n_extreme].copy()
        max_extremes = df_sorted.iloc[-n_extreme:].copy()
        middle = df_sorted.iloc[n_extreme:-n_extreme].copy() if n_extreme > 0 else df_sorted.copy()
        # Optionally, move a fraction of extremes to test set
        if remove_frac_from_extremes > 0:
            min_extreme_test = min_extremes.sample(frac=remove_frac_from_extremes, random_state=random_state)
            max_extreme_test = max_extremes.sample(frac=remove_frac_from_extremes, random_state=random_state)
            min_extremes = min_extremes.drop(index=min_extreme_test.index)
            max_extremes = max_extremes.drop(index=max_extreme_test.index)
            test_df = pd.concat([middle, min_extreme_test, max_extreme_test], ignore_index=True)
        else:
            test_df = middle
        train_df = pd.concat([min_extremes, max_extremes], ignore_index=True)
        # Standardize columns
        train_df = train_df.rename(columns={self.text_col: 'sentence', self.label_col: 'label'})
        test_df = test_df.rename(columns={self.text_col: 'sentence', self.label_col: 'label'})
        self.train_df = train_df
        self.test_df = test_df
        return train_df, test_df

    
    def only_projection_no_split(self):
        self.test_df = self.df.copy()
        self.test_df = self.test_df.rename(columns={self.text_col: 'sentence', self.label_col: 'label'})

    @property
    def texts(self):
        if self.df is None:
            raise ValueError("Data not loaded.")
        return self.df[self.text_col].tolist()

    @property
    def labels(self):
        if self.df is None:
            raise ValueError("Data not loaded.")
        return self.df[self.label_col].tolist()

    @property
    def train_texts(self):
        if self.train_df is None:
            return 
        return self.train_df['sentence'].tolist()

    @property
    def test_texts(self):
        if self.test_df is None:
            raise ValueError("Train/Test split not performed yet.")
        return self.test_df['sentence'].tolist()

    @property
    def train_labels(self):
        if self.train_df is None:
            raise ValueError("Train/Test split not performed yet.")
        return self.train_df['label'].tolist()

    @property
    def test_labels(self):
        if self.test_df is None:
            raise ValueError("Train/Test split not performed yet.")
        return self.test_df['label'].tolist()
