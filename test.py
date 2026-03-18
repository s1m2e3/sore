import re
from sentence_transformers import SentenceTransformer, util

# Good default for semantic similarity on short labels
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

labels = [
    "x is less than 100",
    "x is greater than 100",
    "x is greater than 300",
    "x is equal to 200"
]

def normalize_label(s: str) -> str:
    # split camelCase -> camel Case
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    # underscores/hyphens -> spaces
    s = s.replace("_", " ").replace("-", " ")
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

normalized = [normalize_label(x) for x in labels]
embeddings = model.encode(normalized, convert_to_tensor=True, normalize_embeddings=True)
print(embeddings)
input('yipo')
print("Normalized labels:")
for raw, norm in zip(labels, normalized):
    print(f"{raw!r} -> {norm!r}")

print("\nPairwise cosine similarities:")
for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        score = util.cos_sim(embeddings[i], embeddings[j]).item()
        print(f"{labels[i]:30s} <-> {labels[j]:30s} : {score:.4f}")