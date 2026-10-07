from transformers import pipeline

# Initialize summarization pipeline once
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)

def summarize_to_bullets(text, max_length=80, min_length=30):
    """Summarizes the input text and prints bullet points."""
    result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    summary = result[0]['summary_text']

    bullets = [f"• {sentence.strip().rstrip('.')}" for sentence in summary.split('. ') if sentence.strip()]

    print("\n🔹 Bullet Points:")
    for bullet in bullets:
        print(bullet)


# ✅ Test Case 1
text1 = """
Artificial Intelligence (AI) is rapidly transforming industries from healthcare to finance.
With machine learning and neural networks, computers can now recognize patterns, make decisions,
and even generate human-like content. While this opens up opportunities for automation and efficiency,
experts warn about ethical risks, such as bias in algorithms and job displacement.
"""
summarize_to_bullets(text1)


# ✅ Test Case 2
text2 = """
Over the past decade, streaming services have revolutionized the way people consume media.
Traditional cable subscriptions are in decline, and platforms like Netflix, Disney+, and Amazon Prime
offer on-demand content at lower prices. However, with so many services competing, users are now dealing
with subscription fatigue and rising monthly costs.
"""
summarize_to_bullets(text2)