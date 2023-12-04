import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from nltk import tokenize
import numpy as np
import nltk.data
import statistics

#df = pd.read_csv(r"C:\Users\wccha\Documents\Rutgers\Topics_in_AI\merged_df_Final.csv")
df = pd.read_csv(r"Documents/Topics_in_AI/Final/merged_df_Final.csv")

word_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = GPT2LMHeadModel.from_pretrained("gpt2")

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def calculate_topic_driven_neg_log_likelihood(topic, target): 
  seq = topic + " " + target
  #print("From topic driven seq: ", seq)
  target_encodings = word_tokenizer(target, return_tensors = "pt")
  seq_encodings = word_tokenizer(seq, return_tensors  = "pt")

  max_length = model.config.n_positions
  stride = 1
  target_len = target_encodings.input_ids.size(1)
  #print("target_len: ", target_len)
  seq_len = seq_encodings.input_ids.size(1)
  #print("seq_len: ", seq_len)
  topic_len = seq_len - target_len
  #print("topic_len: ", topic_len)

  nlls = []
  prev_end_loc = 2

  for_loop = 0
  for begin_loc in range(0, seq_len, stride):
      for_loop += 1
      end_loc = min(begin_loc + max_length, seq_len)
      trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
      input_ids = seq_encodings.input_ids[:, begin_loc:end_loc]
      #print("input_ids: ", input_ids)
      target_ids = input_ids.clone()
      target_ids[:, :-trg_len] = -100

      with torch.no_grad():
          outputs = model(input_ids, labels=target_ids)

          neg_log_likelihood = outputs.loss

      nlls.append(neg_log_likelihood)

  ngll_total = 0

  for x in nlls:
      if not x.isnan():
          ngll_total += x.item()

  return ngll_total, len(target)

def calculate_contextual_driven_neg_log_likelihood(topic, target, context):
    seq = topic + context + target
    #print("From context driven seq: ", seq)
    target_encodings = word_tokenizer(target, return_tensors = "pt")
    seq_encodings = word_tokenizer(seq, return_tensors  = "pt")

    max_length = model.config.n_positions
    stride = 1
    target_len = target_encodings.input_ids.size(1)
    #print("target_len: ", target_len)
    seq_len = seq_encodings.input_ids.size(1)
    #print("seq_len: ", seq_len)
    topic_len = seq_len - target_len
    #print("topic_len: ", topic_len)

    nlls = []
    prev_end_loc = 2

    for_loop = 0
    for begin_loc in range(0, seq_len, stride):
        for_loop += 1
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = seq_encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

    ngll_total = 0

    for x in nlls:
        if not x.isnan():
            ngll_total += x.item()

    return ngll_total, len(target)

h = 5
finished_summary_ids = []
final_story_sequentiality = []
#final_story_sequentiality = pd.read_csv(r"C:\Users\wccha\Documents\Rutgers\Topics_in_AI\final_story_sequentiality_df.csv")
# final_story_sequentiality_new_df = pd.read_csv(r"Documents/Topics_in_AI/Final/final_story_sequentiality_df.csv")
# for x in final_story_sequentiality_new_df:
#   final_story_sequentiality.append(x)
# for x in final_story_sequentiality_new_df['recImgPairId']:
#   finished_summary_ids.append(str(x))

#print("starting final_story_sequentiality: ", final_story_sequentiality)
#df = df.iloc[::-1]
  
for i in range(len(df['GPT_Story'])):
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
    #print(id_summary[0])
    topic = df['summary'][i]
    target_story = sentence_tokenizer.tokenize(df['story'][i])
    sentence_sequentiality = []
    for j in range(len(target_story)):

      target = target_story[j]

      topic_driven_nll, target_length = calculate_topic_driven_neg_log_likelihood(topic, target)

      if j >= h:
        #print("j >= h")
        context_sentences = target_story[j-h:j]
        #print("context_sentences: ", context_sentences)
        context = ""
        for sentence in context_sentences:
          context += sentence + " "
        #print("context: ", context)
      else:
        #print("else")
        context_sentences = target_story[0:j]
        #print("context_sentences: ", context_sentences)
        context = ""
        for sentence in context_sentences:
          context += sentence + " "
        #print("context: ", context)

      context_driven_nll, target_length = calculate_contextual_driven_neg_log_likelihood(topic, target, context)
      sentence_sequentiality.append((1/target_length)*(topic_driven_nll - context_driven_nll))

    #   print("topic_driven_nll: ", topic_driven_nll)
    #   print("context_driven_nll", context_driven_nll)
    #   print("sentence_sequentiality: ", sentence_sequentiality)
    
    target_GPT_story = sentence_tokenizer.tokenize(df['GPT_Story'][i])
    GPT_sentence_sequentiality = []
    for j in range(len(target_GPT_story)):
      target_GPT = target_GPT_story[j]

      GPT_topic_driven_nll, GPT_target_length = calculate_topic_driven_neg_log_likelihood(topic, target_GPT)

      if j >= h:
        context_GPT_sentences = target_GPT_story[j-h:j]
        context_GPT = ""
        for sentence in context_GPT_sentences:
          context_GPT += sentence + " "
      else:
        context_GPT_sentences = target_GPT_story[0:j]
        context_GPT = ""
        for sentence in context_GPT_sentences:
          context_GPT += sentence + " "

      GPT_context_driven_nll, GPT_target_length = calculate_contextual_driven_neg_log_likelihood(topic, target_GPT, context_GPT)
      GPT_sentence_sequentiality.append((1/GPT_target_length)*(GPT_topic_driven_nll - GPT_context_driven_nll))

    #   print("GPT_topic_driven_nll: ", GPT_topic_driven_nll)
    #   print("GPT_context_driven_nll", GPT_context_driven_nll)
    #   print("GPT_sentence_sequentiality: ", GPT_sentence_sequentiality)

    story_sequentiality = statistics.mean(sentence_sequentiality)
    #print("story_sequentiality: ", story_sequentiality)
    GPT_story_sequentiality = statistics.mean(GPT_sentence_sequentiality)
    # print("GPT_story_sequentiality: ", GPT_story_sequentiality)
    id_summary.append(story_sequentiality)
    id_summary.append(GPT_story_sequentiality)
    final_story_sequentiality.append(id_summary)
    finished_summary_ids.append(id_summary[0])
    #print(final_story_sequentiality)
    final_story_sequentiality_df = pd.DataFrame(final_story_sequentiality)
    final_story_sequentiality_df.rename(columns={0: "recImgPairId"}, inplace=True)
    final_story_sequentiality_df.rename(columns={1: "Human_Story_Sequentiality"}, inplace=True)
    final_story_sequentiality_df.rename(columns={2: "GPT_Story_Sequentiality"}, inplace=True)
    final_story_sequentiality_df.to_csv(r"Documents/Topics_in_AI/Final/final_story_sequentiality.csv")
    #final_story_sequentiality_df.to_csv(r"C:\Users\wccha\Documents\Rutgers\Topics_in_AI\backwards_final_story_sequentiality_df.csv")
    print(final_story_sequentiality_df)

