import pandas as pd
import re

dataset = pd.read_csv(r'../data/bug_reports_filtered.csv')


def clean_data(text:str):
    """
    Clean bug report text.
    """
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

if __name__ == "__main__":
    dataset['clean_text'] = dataset['text'].apply(clean_data)

    dataset.to_csv( "../data/bug_reports_clean.csv", index=False )

    print("Processed Dataset Saved")
    print(dataset.head())