import pandas as pd

df = pd.read_csv(r"~/Documents/Topics_in_AI/Final/Cropped_Ultimate_final_data.csv")

final_GPT = []
final_ids_GPT = []
final_Human = []
final_ids_Human = []

for x in range(len(df)):
  if df["recImgPairId"][x] not in final_ids_GPT:
    temp = []
    temp.append(df["GPT_Story_Sequentiality"][x])
    temp.append(df["GPT_story_average_sentence_length"][x])
    temp.append(df["GPT_story_concreteness_mean"][x])
    temp.append(df["GPT_story_concreteness_lower"][x])
    temp.append(df["GPT_story_concreteness_upper"][x])
    temp.append(df["GPT_story_sentiment_mean"][x])
    temp.append(1)
    final_GPT.append(temp)
    final_ids_GPT.append(df["recImgPairId"][x])


final_GPT_df = pd.DataFrame(final_GPT)
final_GPT_df.rename(columns={0: "story_sequentiality"}, inplace=True)
final_GPT_df.rename(columns={1: "story_average_sentence_length"}, inplace=True)
final_GPT_df.rename(columns={2: "story_concreteness_mean"}, inplace=True)
final_GPT_df.rename(columns={3: "story_concreteness_lower"}, inplace=True)
final_GPT_df.rename(columns={4: "story_concreteness_upper"}, inplace=True)
final_GPT_df.rename(columns={5: "story_sentiment_mean"}, inplace=True)
final_GPT_df.rename(columns={6: "is_gpt"}, inplace=True)

final = []
final_ids = []
for x in range(len(df)):
  if df["recImgPairId"][x] not in final_ids:
    temp = []
    temp.append(df["Human_Story_Sequentiality"][x])
    temp.append(df["story_average_sentence_length"][x])
    temp.append(df["story_concreteness_mean"][x])
    temp.append(df["story_concreteness_lower"][x])
    temp.append(df["story_concreteness_upper"][x])
    temp.append(df["story_sentiment_mean"][x])
    temp.append(0)
    final.append(temp)
    final_ids.append(df["recImgPairId"][x])

final = pd.DataFrame(final)
final.rename(columns={0: "story_sequentiality"}, inplace=True)
final.rename(columns={1: "story_average_sentence_length"}, inplace=True)
final.rename(columns={2: "story_concreteness_mean"}, inplace=True)
final.rename(columns={3: "story_concreteness_lower"}, inplace=True)
final.rename(columns={4: "story_concreteness_upper"}, inplace=True)
final.rename(columns={5: "story_sentiment_mean"}, inplace=True)
final.rename(columns={6: "is_gpt"}, inplace=True)

ML_df = pd.concat([final_GPT_df, final])



ML_df.to_csv("~/Documents/Topics_in_AI/Final/ML_final_data.csv")