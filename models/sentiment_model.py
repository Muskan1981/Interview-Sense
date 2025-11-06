from transformers import pipeline

analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

def analyze_sentiment(text):
    text = text.strip()
    if not text:
        return {"label": "NEUTRAL", "score": 0.5}
    result = analyzer(text[:512])[0]
    return {
        "label": result["label"].capitalize(),
        "score": round(result["score"], 2)
    }