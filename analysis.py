"""
Author: Sobhan
Date: 23-05-2025
Project: Bones Ltd. job technical challenge. 
File: anaylis.py
Description and Usage:
   The functions in this file compute the sentiments and filler word
   ratios.
"""

from transformers import pipeline
import re
import spacy


def compute_filler_ratio(text, filler_words):
   '''Find the filler_words in a text and returns the ratio of filler_words/all_words
    text: the intial text to look at. 
    filler_words: All filler words we consider for this function.
    '''
   text = text.strip()

   '''
   # only one of these two should be used. The first one in 1)
   # cannot capture fillers like 'I swear' or 'You know', but
   # spacy is powerful enough to help capture them in text.
   # BTW, Spacy is slower! Uncomment if needed for faster one in 1). 

   # 1)
   all_words = text.lower().split(' ')

   filler_count = 0
   for word in all_words:
      if(word in filler_words or word in filler_words_with_comma):
         filler_count+=1
   filler_words_ratio = filler_count/len(all_words)
   return filler_words_ratio 
   '''

   # 2) using spacy tool
   # Load spaCy model
   nlp = spacy.load("en_core_web_sm")
   doc = nlp(text.lower())

   # Count total words (excluding punctuation, digits)
   total_words = len([token for token in doc if token.is_alpha])

   # Count filler words (both single and multi-word) --> real power of spacy
   filler_count = 0
   for phrase in filler_words:
      pattern = r'\b' + re.escape(phrase.lower()) + r'\b'
      matches = re.findall(pattern, text.lower())
      filler_count += len(matches)

   # Calculate ratio
   filler_words_ratio = filler_count / total_words if total_words > 0 else 0
   return filler_words_ratio


def compute_sentiment(conversation, filler_words):
    # Load Hugging Face sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")

    conversation_lines = conversation.splitlines()
    
    # compute the sentiments for all lines in the conversation  
    results = []
    for line in conversation_lines:
        # l = f.read()
        print(line)
        speaker, msg = line.split(':', 1)
        filler_words_ratio = compute_filler_ratio(text=msg,
                                                  filler_words=filler_words)

        sentiment = sentiment_pipeline(msg)[0]
        results.append({
            "speaker": speaker,
            "message": msg,
            "sentiment": sentiment['label'],
            "score": sentiment['score'],
            "filler_words_ratio": filler_words_ratio,

        })
        
    return results
