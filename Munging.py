import pandas as pd

df_length = pd.read_csv("~/Documents/Topics_in_AI/Final/final_story_sentence_length_df.csv")
df_concret = pd.read_csv("~/Documents/Topics_in_AI/Final/final_story_concreteness.csv")
df_sentiment = pd.read_csv("~/Documents/Topics_in_AI/Final/final_story_sentiment_df.csv")
complete_df = pd.read_csv("~/Documents/Topics_in_AI/Final/Complete_data.csv")
final_df = pd.merge(complete_df, df_length, on = "recImgPairId", how = "left")
final_df = pd.merge(final_df, df_concret, on = "recImgPairId", how = "left")
final_df = pd.merge(final_df, df_sentiment, on = "recImgPairId", how = "right")
#final_df.drop_duplicates()
final_df.to_csv("~/Documents/Topics_in_AI/Final/Ultimate_final_data.csv")