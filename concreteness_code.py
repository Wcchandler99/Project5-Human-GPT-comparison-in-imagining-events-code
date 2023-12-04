import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from nltk import tokenize
import numpy as np
import nltk.data
import statistics

df = pd.read_csv(r"Documents/Topics_in_AI/Final/merged_df_Final.csv")

concreteness_ratings = pd.read_csv(r"Documents/Topics_in_AI/Final/concreteness_ratings.csv")
#concreteness_ratings = pd.read_csv(r"C:\Users\wccha\Documents\Rutgers\Topics_in_AI\concreteness_ratings.csv")
finished_summary_ids = []
final_story_concreteness = []
#final_story_concreteness = pd.read_csv(r"C:\Users\wccha\Documents\Rutgers\Topics_in_AI\final_story_concreteness_df.csv")
#final_story_concreteness = pd.read_csv(r"Documents/Topics_in_AI/Final/final_story_concreteness_df.csv")
# for x in final_story_concreteness['recImgPairId']:
#   finished_summary_ids.append(str(x))

#print("starting final_story_concreteness: ", final_story_concreteness)
  
for i in range(len(df['summary'])):
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
    story_concreteness_mean_ratings = []
    story_concreteness_lower_ratings = []
    story_concreteness_upper_ratings = []
    #print("df['GPT_Story'][i]: ", type(df['GPT_Story'][i]))
    for j in range(len(concreteness_ratings['Word'])):
      #print("type(concreteness_ratings['Word'][j]): ", type(concreteness_ratings['Word'][j]))
      if type(concreteness_ratings['Word'][j]) == str:
        if concreteness_ratings['Word'][j] in df['story'][i]:
            story_concreteness_mean_ratings.append(concreteness_ratings['Conc.M'][j])
            story_concreteness_lower_ratings.append(concreteness_ratings['lower_bounds'][j])
            story_concreteness_upper_ratings.append(concreteness_ratings['upper_bound'][j])
    
    GPT_story_concreteness_mean_ratings = []
    GPT_story_concreteness_lower_ratings = []
    GPT_story_concreteness_upper_ratings = []
    for j in range(len(concreteness_ratings['Word'])):
      #print("type(concreteness_ratings['Word'][j]): ", type(concreteness_ratings['Word'][j]))
      if type(concreteness_ratings['Word'][j]) == str:
        if concreteness_ratings['Word'][j] in df['GPT_Story'][i]:
            #print(concreteness_ratings['Word'][j])
            GPT_story_concreteness_mean_ratings.append(concreteness_ratings['Conc.M'][j])
            GPT_story_concreteness_lower_ratings.append(concreteness_ratings['lower_bounds'][j])
            GPT_story_concreteness_upper_ratings.append(concreteness_ratings['upper_bound'][j])

    story_concreteness_mean = statistics.mean(story_concreteness_mean_ratings)
    story_concreteness_lower = statistics.mean(story_concreteness_lower_ratings)
    story_concreteness_upper = statistics.mean(story_concreteness_upper_ratings)
    GPT_story_concreteness_mean = statistics.mean(GPT_story_concreteness_mean_ratings)
    GPT_story_concreteness_lower = statistics.mean(GPT_story_concreteness_lower_ratings)
    GPT_story_concreteness_upper = statistics.mean(GPT_story_concreteness_upper_ratings)
    #print("story_concreteness_mean: ", story_concreteness_mean)
    #print("GPT_story_concreteness_mean: ", GPT_story_concreteness_mean)
    id_summary.append(story_concreteness_mean)
    id_summary.append(story_concreteness_lower)
    id_summary.append(story_concreteness_upper)
    id_summary.append(GPT_story_concreteness_mean)
    id_summary.append(GPT_story_concreteness_lower)
    id_summary.append(GPT_story_concreteness_upper)
    final_story_concreteness.append(id_summary)
    finished_summary_ids.append(id_summary[0])
    #print(final_story_concreteness)
    #print(id_summary)
    final_story_concreteness_df = pd.DataFrame(final_story_concreteness)
    final_story_concreteness_df.rename(columns={0: "recImgPairId"}, inplace=True)
    final_story_concreteness_df.rename(columns={1: "story_concreteness_mean"}, inplace=True)
    final_story_concreteness_df.rename(columns={2: "story_concreteness_lower"}, inplace=True)
    final_story_concreteness_df.rename(columns={3: "story_concreteness_upper"}, inplace=True)
    final_story_concreteness_df.rename(columns={4: "GPT_story_concreteness_mean"}, inplace=True)
    final_story_concreteness_df.rename(columns={5: "GPT_story_concreteness_lower"}, inplace=True)
    final_story_concreteness_df.rename(columns={6: "GPT_story_concreteness_upper"}, inplace=True)
    final_story_concreteness_df.to_csv(r"Documents/Topics_in_AI/Final/final_story_concreteness.csv")
    print(final_story_concreteness_df)
#final_story_concreteness_df.to_csv(r"C:\Users\wccha\Documents\Rutgers\Topics_in_AI\final_story_sequentiality_df.csv")
print("DONE")
