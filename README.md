<h1 align="center">SemanticProjection</h1>
Beta release of the Lightweight Semantic Projection Package.

# Installation
You can install `SemanticProjecter` via [pip] from the [git]:
```bash
pip install git+"https://github.com/lauritswl/SemanticProjection"
```

# Usage
See the following code snippet for an example of how to project and standardize text snippets with the SemanticProjector Class:
```python
from semanticprojection.SemanticProjecter import SemanticProjector
# Define text as list:
list_of_texts = ["This is negative.", "This is positive.", "This is neutral", "I hate you!", "I like this so much!", "They didn't care for it"]
# Create an projector object from SemanticProjector class.
projector = SemanticProjector()
# Project the text, by using the Sentiment Vector:
projector.project_texts(
    texts=list_of_texts,
    concept_vector="Sentiment"
)
# Add a standardized collumn - adds interpretability in larger datasets
projector.standardize()

# Show results (stored in class atribute- self.results)
print(projector.results)
```
 Vector Lookup

| Title             | Description                                      |
|------------------|--------------------------------------------------|
| "Sentiment"  | Encodes the sentiment of a sentence (negative → positive). |
| WIP               | Working on adding more of the shelf vectors         |


# 
