import re
import string

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+|\#", "", text)         # remove mentions/hashtags
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)             # remove numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataframe(df, text_column):
    df = df.copy()
    df["clean_text"] = df[text_column].astype(str).apply(clean_text)
    return df
