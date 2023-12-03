import pandas as pd


final = []
final_ids = []
df = pd.read_csv(r"~/Documents/Topics_in_AI/Final/Ultimate_final_data.csv")
for x in range(len(df)):
  if df["recImgPairId"][x] not in final_ids:
    temp = []
    temp.append(df["recImgPairId"][x])
    temp.append(df["summary"][x])
    temp.append(df["story"][x])
    temp.append(df["GPT_Story"][x])
    temp.append(df["Human_Story_Sequentiality"][x])
    temp.append(df["GPT_Story_Sequentiality"][x])
    temp.append(df["story_average_sentence_length"][x])
    temp.append(df["GPT_story_average_sentence_length"][x])
    temp.append(df["story_concreteness_mean"][x])
    temp.append(df["story_concreteness_lower"][x])
    temp.append(df["story_concreteness_upper"][x])
    temp.append(df["GPT_story_concreteness_mean"][x])
    temp.append(df["GPT_story_concreteness_lower"][x])
    temp.append(df["GPT_story_concreteness_upper"][x])
    temp.append(df["summary_sentiment_mean"][x])
    temp.append(df["story_sentiment_mean"][x])
    temp.append(df["GPT_story_sentiment_mean"][x])
    temp.append(df["story_summary_sentiment_difference"][x])
    temp.append(df["GPT_story_summary_sentiment_difference"][x])
    final.append(temp)
    final_ids.append(df["recImgPairId"][x])


final_df = pd.DataFrame(final)
final_df.rename(columns={0: "recImgPairId"}, inplace=True)
final_df.rename(columns={1: "summary"}, inplace=True)
final_df.rename(columns={2: "story"}, inplace=True)
final_df.rename(columns={3: "GPT_Story"}, inplace=True)
final_df.rename(columns={4: "Human_Story_Sequentiality"}, inplace=True)
final_df.rename(columns={5: "GPT_Story_Sequentiality"}, inplace=True)
final_df.rename(columns={6: "story_average_sentence_length"}, inplace=True)
final_df.rename(columns={7: "GPT_story_average_sentence_length"}, inplace=True)
final_df.rename(columns={8: "story_concreteness_mean"}, inplace=True)
final_df.rename(columns={9: "story_concreteness_lower"}, inplace=True)
final_df.rename(columns={10: "story_concreteness_upper"}, inplace=True)
final_df.rename(columns={11: "GPT_story_concreteness_mean"}, inplace=True)
final_df.rename(columns={12: "GPT_story_concreteness_lower"}, inplace=True)
final_df.rename(columns={13: "GPT_story_concreteness_upper"}, inplace=True)
final_df.rename(columns={14: "summary_sentiment_mean"}, inplace=True)
final_df.rename(columns={15: "story_sentiment_mean"}, inplace=True)
final_df.rename(columns={16: "GPT_story_sentiment_mean"}, inplace=True)
final_df.rename(columns={17: "story_summary_sentiment_dffierence"}, inplace=True)
final_df.rename(columns={18: "GPT_story_summary_sentiment_dfference"}, inplace=True)

final_df.to_csv("~/Documents/Topics_in_AI/Final/Cropped_Ultimate_final_data.csv")