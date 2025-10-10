from src.SemanticProjection.SemanticProjecter import SemanticProjector
def TextToVector(texts, concept_vector):
    Projector=SemanticProjector()
    return Projector.project_texts(texts=texts, concept_vector=concept_vector)

if __name__ == "__main__":
    df =TextToVector(
        texts=["I love programming", "I hate bugs", "She is very sexy", "She is very ugly"],
        concept_vector="Sentiment")
    print(df)
