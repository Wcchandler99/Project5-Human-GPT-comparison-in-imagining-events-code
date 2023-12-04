import pandas as pd
import numpy as np
import statistics

df = pd.read_csv(r"Documents/Topics_in_AI/Final/merged_df_Final.csv")
#df = pd.read_csv(r"C:\Users\wccha\Documents\Rutgers\Topics_in_AI\merged_df_Final.csv")
#sentiment_ratings = pd.read_csv(r"C:/Users/wccha/Documents/Rutgers/Topics_in_AI/SentiWordNet-master/data/SentiWordNet_3.0.0.txt", sep = '\t')
sentiment_ratings = pd.read_csv(r"Documents/Topics_in_AI/Final/SentiWordNet_3.0.0.txt", sep = '\t')
sentiment_ratings['score'] = sentiment_ratings['PosScore'] - sentiment_ratings['NegScore']
sentiment_ratings['Word'] = range(117660)
for i in range(len(sentiment_ratings['SynsetTerms'])):
    sentiment_ratings['Word'][i] = sentiment_ratings['SynsetTerms'][i].split('#')[0]
finished_summary_ids = []
final_story_sentiment = []

#final_story_sentiment_new_df = pd.read_csv(r"C:\Users\wccha\Documents\Rutgers\Topics_in_AI\final_story_sentiment_df.csv")
final_story_sentiment_new_df = pd.read_csv(r"Documents/Topics_in_AI/Final/final_story_sentiment_df.csv")

for x in final_story_sentiment_new_df['recImgPairId']:
  finished_summary_ids.append(str(x))

for x in range(1, len(final_story_sentiment_new_df)):
  temp = []
  temp.append(final_story_sentiment_new_df["recImgPairId"][x])
  temp.append(final_story_sentiment_new_df["summary_sentiment_mean"][x])
  temp.append(final_story_sentiment_new_df["story_sentiment_mean"][x])
  temp.append(final_story_sentiment_new_df["GPT_story_sentiment_mean"][x])
  temp.append(final_story_sentiment_new_df["story_summary_sentiment_difference"][x])
  temp.append(final_story_sentiment_new_df["GPT_story_summary_sentiment_difference"][x])
  final_story_sentiment.append(temp)

#print("starting final_story_sentiment: ", final_story_sentiment)
  
for i in range(670, len(df['summary'])):
#for i in range(3):
  id_summary = []
  if df['recImgPairId'][i] and pd.notnull(df['recImgPairId'][i]):
    id_summary.append(df['recImgPairId'][i])
  elif df['recAgnPairId'][i] and pd.notnull(df['recAgnPairId'][i]):
    df['recImgPairId'][i] = df['recAgnPairId'][i]
    id_summary.append(df['recAgnPairId'][i])
  else:
    id_summary.append('NAPairId')
  if id_summary[0] not in finished_summary_ids:
    summary_sentiment_mean_ratings = []
    #print("df['GPT_Story'][i]: ", type(df['GPT_Story'][i]))
    for j in range(len(sentiment_ratings['Word'])):
      #print("type(sentiment_ratings['score'][j]): ", type(sentiment_ratings['score'][j]))
      if type(sentiment_ratings['Word'][j]) == str:
        if sentiment_ratings['Word'][j] in df['summary'][i]:
            if sentiment_ratings['score'][j] != 0:
                if not np.isnan(sentiment_ratings['score'][j]):
                    summary_sentiment_mean_ratings.append(sentiment_ratings['score'][j])
                    
    story_sentiment_mean_ratings = []
    #print("df['GPT_Story'][i]: ", type(df['GPT_Story'][i]))
    for j in range(len(sentiment_ratings['Word'])):
      #print("type(sentiment_ratings['score'][j]): ", type(sentiment_ratings['score'][j]))
      if type(sentiment_ratings['Word'][j]) == str:
        if sentiment_ratings['Word'][j] in df['story'][i]:
            if sentiment_ratings['score'][j] != 0:
                if not np.isnan(sentiment_ratings['score'][j]):
                    story_sentiment_mean_ratings.append(sentiment_ratings['score'][j])
    
    GPT_story_sentiment_mean_ratings = []
    for j in range(len(sentiment_ratings['Word'])):
      #print("type(sentiment_ratings['Word'][j]): ", type(sentiment_ratings['Word'][j]))
      if type(sentiment_ratings['Word'][j]) == str:
        if sentiment_ratings['Word'][j] in df['GPT_Story'][i]:
            if sentiment_ratings['score'][j] != 0:
                if not np.isnan(sentiment_ratings['score'][j]):
                    #print(sentiment_ratings['Word'][j])
                    GPT_story_sentiment_mean_ratings.append(sentiment_ratings['score'][j])

    
    #print("story_sentiment_mean_ratings: ", story_sentiment_mean_ratings)
    #print("GPT_story_sentiment_mean_ratings: ", GPT_story_sentiment_mean_ratings)
    if summary_sentiment_mean_ratings:
        summary_sentiment_mean = statistics.mean(summary_sentiment_mean_ratings)
    if story_sentiment_mean_ratings:
        story_sentiment_mean = statistics.mean(story_sentiment_mean_ratings)
    if GPT_story_sentiment_mean_ratings:
        GPT_story_sentiment_mean = statistics.mean(GPT_story_sentiment_mean_ratings)
    story_summary_sentiment_difference = abs(summary_sentiment_mean) - abs(story_sentiment_mean)
    GPT_story_summary_sentiment_difference = abs(summary_sentiment_mean) - abs(GPT_story_sentiment_mean)
    print("summary_sentiment_mean: ", summary_sentiment_mean)
    print("story_sentiment_mean: ", story_sentiment_mean)
    print("GPT_story_sentiment_mean: ", GPT_story_sentiment_mean)
    id_summary.append(summary_sentiment_mean)
    id_summary.append(story_sentiment_mean)
    id_summary.append(GPT_story_sentiment_mean)
    id_summary.append(story_summary_sentiment_difference)
    id_summary.append(GPT_story_summary_sentiment_difference)
    final_story_sentiment.append(id_summary)
    finished_summary_ids.append(id_summary[0])
    #print(final_story_sentiment)
    final_story_sentiment_df = pd.DataFrame(final_story_sentiment)
    final_story_sentiment_df.rename(columns={0: "recImgPairId"}, inplace=True)
    final_story_sentiment_df.rename(columns={1: "summary_sentiment_mean"}, inplace=True)
    final_story_sentiment_df.rename(columns={2: "story_sentiment_mean"}, inplace=True)
    final_story_sentiment_df.rename(columns={3: "GPT_story_sentiment_mean"}, inplace=True)
    final_story_sentiment_df.rename(columns={4: "story_summary_sentiment_difference"}, inplace=True)
    final_story_sentiment_df.rename(columns={5: "GPT_story_summary_sentiment_difference"}, inplace=True)
    final_story_sentiment_df.to_csv(r"Documents/Topics_in_AI/Final/final_story_sentiment_df.csv")
    #final_story_sentiment_df.to_csv(r"C:\Users\wccha\Documents\Rutgers\Topics_in_AI\final_story_sentiment_df.csv")
    print(final_story_sentiment_df)
print("Finished")


