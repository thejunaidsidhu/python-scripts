# If the text is written in formal, Imformal or Causal Tone



# Step 2: Import libraries
from transformers import pipeline

# Step 3: Load zero-shot classification pipeline for tone detection
tone_classifier = pipeline("zero-shot-classification",
                           model="facebook/bart-large-mnli",
                           device=-1)  # use CPU

# Step 4: Tone detection function
def detect_tone(text):
    candidate_labels = ["Formal", "Informal", "Neutral"]
    result = tone_classifier(text, candidate_labels)
    scores = {label: round(score, 3) for label, score in zip(result["labels"], result["scores"])}
    predicted_label = result["labels"][0]
    return predicted_label, scores

# Step 5: Example student answers
student_answers = [
    "Dear Sir, I believe the battery efficiency has not improved due to chemistry limits.",
    "Yo! I think the battery ain't better because of the stuff it's made from.",
    "Solid-state batteries are cool but not ready yet, I guess.",
    "Respectfully, the answer lies in lab conditions and fast charging issues.",
    "Battery's still meh because solid-state ones ain't perfect yet.",
    "According to recent developments, energy density is still an issue.",
    "Tbh, they haven't figured out how to make 'em work outside labs.",
    "As per the research, battery life is limited due to material fragility.",
    "Not ready yet cuz they charge slow and break easy lol.",
    "The chemistry hasn't evolved as fast as processors, so batteries lag."
]

# Step 6: Run tone detection for all students
print("STUDENT |     TONE     |       SCORES")
print("----------------------------------------------")
for i, answer in enumerate(student_answers, 1):
    tone, scores = detect_tone(answer)
    print(f"   {i:<5} | {tone:<11} | {scores}")
