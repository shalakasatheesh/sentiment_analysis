import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download('all')

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
from scipy.special import softmax

plt.style.use('ggplot')

def read_data(file_path='Reviews.csv'):
    """
    Function to read input csv file and
    output a pd.DataFrame
    """
    df = pd.read_csv(file_path)
    df = df.head(500)
    return df

def polarity_scores_roberta(example): 
    """
    Determine the polarity scores using RoBerta
    """
    # Initialise RoBerta Model
    pretrained_model = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokeniser = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)

    # Compute results
    encoded_text = tokeniser(example, return_tensors="pt")
    output = model(**encoded_text)
    berta_scores = output[0][0].detach().numpy()
    berta_scores = softmax(berta_scores)
    berta_score_dict = {
        'berta_neg': berta_scores[0],
        'berta_neu': berta_scores[1],
        'berta_pos': berta_scores[2]
    } 
    return berta_score_dict

def main():
    # Load data
    df = read_data()

    # Initialise VADER model
    analyser = SentimentIntensityAnalyzer()

    # Compute results 
    berta_results = {}
    vaders_results = {}
    results = {}
    for i, data in tqdm(df.iterrows(), total=len(df)):
        try:
            text = data["Text"]
            text_id = data["Id"]

            vaders_results = analyser.polarity_scores(text)
            vader_results_final={}

            for key, value in vaders_results.items():
                vader_results_final[f"vader_{key}"] = value

            berta_results = polarity_scores_roberta(text)

            both = {**vader_results_final, **berta_results}

            results[text_id] = both
        except:
            print(f"Broke for id {text_id}")

    results_df = pd.DataFrame(results).T
    results_df = results_df.reset_index().rename(columns={'index': 'Id'})
    results_df = results_df.merge(df, how='left')
    results_df.to_pickle("results_df.pkl")


main()