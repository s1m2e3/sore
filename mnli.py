from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

labels = ["Temperature","Temp_DegreesCelsius","Temp_DegreesFahrenheit"
]
MODEL_NAME = "facebook/bart-large-mnli"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# bart-large-mnli uses 3 labels:
# contradiction, neutral, entailment
id2label = model.config.id2label
print("Model labels:", id2label)

def nli(premise: str, hypothesis: str):
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = F.softmax(logits, dim=-1)

    return {
        id2label[i]: float(probs[i])
        for i in range(len(probs))
    }

# Run pairwise NLI in both directions
for i, premise in enumerate(labels):
    for j, hypothesis in enumerate(labels):
        if i == j:
            continue

        probs = nli(premise, hypothesis)
        best_label = max(probs, key=probs.get)

        print("=" * 80)
        print(f"Premise   : {premise}")
        print(f"Hypothesis: {hypothesis}")
        print(f"Prediction: {best_label}")
        print(
            "Scores    : "
            f"contradiction={probs.get('contradiction', 0):.4f}, "
            f"neutral={probs.get('neutral', 0):.4f}, "
            f"entailment={probs.get('entailment', 0):.4f}"
        )