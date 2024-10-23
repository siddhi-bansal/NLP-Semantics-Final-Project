# Import the necessary modules
from laserembeddings import Laser

# Initialize LASER
laser = Laser()

# List of sentences (these can be in different languages)
sentences = [
    "This is a sentence in English.",
    "C'est une phrase en français.",
    "这是中文的一句话。",
    "Dies ist ein Satz auf Deutsch.",
    "यह हिंदी में एक वाक्य है।"
]

# Generate sentence embeddings
embeddings = laser.embed_sentences(sentences, lang='en')  # 'en' here is the fallback language

# Print the embeddings
for i, emb in enumerate(embeddings):
    print(f"Sentence {i+1}: {sentences[i]}")
    print(f"Embedding: {emb}\n")

# Do cosine similarity on the english and hindi sentence embeddings
from sklearn.metrics.pairwise import cosine_similarity

# English sentence embedding
en_emb = embeddings[0]

# Hindi sentence embedding
hi_emb = embeddings[-1]

# Calculate the cosine similarity
similarity = cosine_similarity([en_emb], [hi_emb])
print(f"Cosine similarity between the English and Hindi sentences: {similarity[0][0]}")

# With German now
ger_emb = embeddings[-2]

similarity = cosine_similarity([en_emb], [ger_emb])
print(f"Cosine similarity between the English and German sentences: {similarity[0][0]}")