import numpy as np

def analyze_sentiment(text):
    """Analyze the sentiment of movie review text.
    Returns a dictionary with 'label' (Positive, Neutral, Negative) and 'score' (0 to 100)."""
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return {"label": "Neutral", "score": 50.0}
        
    text_lower = text.lower()
    
    # Positives list
    positives = [
        "love", "amazing", "great", "excellent", "masterpiece", "beautiful", "wonderful", "fantastic",
        "good", "liked", "favourite", "favorite", "must watch", "brilliant", "perfect", "enjoyed",
        "awesome", "best", "superb", "incredible", "stunning", "gripping", "classic", "outstanding",
        "gem", "thrilling", "highly recommend", "masterful", "entertaining"
    ]
    
    # Negatives list
    negatives = [
        "boring", "waste", "terrible", "worst", "bad", "disappointed", "disappointing", "hate",
        "awful", "stupid", "annoying", "poor", "slow", "pointless", "weak", "hated", "crap",
        "mediocre", "dull", "lacked", "frustrated", "frustrating", "flawed", "ridiculous", "dumb",
        "worst movie", "garbage", "fail", "mess", "unwatchable"
    ]
    
    # Count occurrences
    pos_count = sum(text_lower.count(word) for word in positives)
    neg_count = sum(text_lower.count(word) for word in negatives)
    
    total = pos_count + neg_count
    
    if total == 0:
        # Default neutral
        return {"label": "Neutral", "score": 50.0}
        
    # Calculate score from -1.0 to 1.0
    sentiment_score = (pos_count - neg_count) / total
    
    # Convert to 0% - 100% scale
    percentage = float(np.round(((sentiment_score + 1.0) / 2.0) * 100.0, 1))
    
    if sentiment_score > 0.2:
        label = "Positive"
    elif sentiment_score < -0.2:
        label = "Negative"
    else:
        label = "Neutral"
        
    return {"label": label, "score": percentage}
