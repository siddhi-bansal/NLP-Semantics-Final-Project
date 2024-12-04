from sentence_transformers import SentenceTransformer
sentences = ["Hello", "Hola", "नमस्ते"]

model = SentenceTransformer('sentence-transformers/LaBSE')
embeddings = model.encode(sentences)

# english, spanish, hindi
# do pairwise cosine similarity between all 3

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print(cosine_similarity(np.array([embeddings[0]]), np.array([embeddings[1]])))
print(cosine_similarity(np.array([embeddings[0]]), np.array([embeddings[2]])))
print(cosine_similarity(np.array([embeddings[1]]), np.array([embeddings[2]])))