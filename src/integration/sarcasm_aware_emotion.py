POSITIVE_EMOTIONS = {
    "joy",
    "love",
    "excitement",
    "optimism/approval"
}

NEGATIVE_EMOTIONS = {
    "anger",
    "disappointment",
    "sadness",
    "disgust",
    "confusion"
}


def adjust_emotion(predicted_emotion, sarcasm_label):
    """
    Adjust emotion prediction based on sarcasm detection.
    """
    if sarcasm_label == "sarcastic" and predicted_emotion in POSITIVE_EMOTIONS:
        return "disappointment"
    return predicted_emotion
