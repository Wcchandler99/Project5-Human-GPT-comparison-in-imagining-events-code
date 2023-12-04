import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk import tokenize
import numpy as np
import nltk.data
import statistics

df = pd.read_csv(r"Documents/Topics_in_AI/Final/merged_df_Final.csv")

word_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = GPT2LMHeadModel.from_pretrained("gpt2")

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

finished_summary_ids = []
final_story_sentence_length = []
#final_story_sentence_length = pd.read_csv(r"C:\Users\wccha\Documents\Rutgers\Topics_in_AI\final_story_sentence_length_df.csv")
#final_story_sentence_length = pd.read_csv(r"Documents/Topics_in_AI/Final/final_story_sentence_length_df.csv")
# for x in final_story_concreteness['recImgPairId']:
#   finished_summary_ids.append(str(x))

print("starting final_story_concreteness: ", final_story_sentence_length)
  
for i in range(len(df['summary'])):
#for i in range(2):
  id_summary = []
  if df['recImgPairId'][i] and pd.notnull(df['recImgPairId'][i]):
    id_summary.append(df['recImgPairId'][i])
  elif df['recAgnPairId'][i] and pd.notnull(df['recAgnPairId'][i]):
    df['recImgPairId'][i] = df['recAgnPairId'][i]
    id_summary.append(df['recAgnPairId'][i])
  else:
    id_summary.append('NAPairId')
  if id_summary[0] not in finished_summary_ids:
    GPT_story_sentence_lengths = []
    story_sentence_lengths = []
    GPT_story_sentences = sentence_tokenizer.tokenize(df['GPT_Story'][i])
    story_sentences = sentence_tokenizer.tokenize(df['story'][i])
    for sentence in GPT_story_sentences:
      GPT_story_sentence_words = word_tokenizer.tokenize(sentence)
      GPT_story_sentence_lengths.append(len(GPT_story_sentence_words))
    for sentence in story_sentences:
      story_sentence_words = word_tokenizer.tokenize(sentence)
      story_sentence_lengths.append(len(story_sentence_words))


    GPT_story_average_sentence_length = statistics.mean(GPT_story_sentence_lengths)
    story_average_sentence_length = statistics.mean(story_sentence_lengths)
    id_summary.append(story_average_sentence_length)
    id_summary.append(GPT_story_average_sentence_length)
    final_story_sentence_length.append(id_summary)
    finished_summary_ids.append(id_summary[0])
    #print(final_story_concreteness)
    final_story_sentence_length_df = pd.DataFrame(final_story_sentence_length)
    final_story_sentence_length_df.rename(columns={0: "recImgPairId"}, inplace=True)
    final_story_sentence_length_df.rename(columns={1: "story_average_sentence_length"}, inplace=True)
    final_story_sentence_length_df.rename(columns={2: "GPT_story_average_sentence_length"}, inplace=True)
    print(final_story_sentence_length_df)
    final_story_sentence_length_df.to_csv(r"Documents/Topics_in_AI/Final/final_story_sentence_length_df.csv")

