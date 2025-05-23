"""
Author: Sobhan
Date: 23-05-2025
Project: Bones Ltd. job technical challenge. 
File: app.py
Description and Usage:
   This file includes the main() function and is actually the first
   file that is needed to be run. suggestion run: > streamlit run app.py
"""


import configparser
from utils import (read_fillers_from_file,
                   load_conversation,
                   render_and_visualize)

from analysis import compute_sentiment


def main(transcript_file='transcript.txt',
         filler_words_file='filler_words.txt',
         transcript_auto_fill=False):

    # loading all fillers
    all_filler_words = read_fillers_from_file(filler_words_file)

    # loading conversation 
    conversation = load_conversation(auto_fill=transcript_auto_fill,
                                     file_path=transcript_file)

    # compute sentiments
    results = compute_sentiment(conversation, all_filler_words)

    # display with conversation streamlit and plot graphs
    render_and_visualize(inputs=results)


if __name__ == "__main__":

    # Load config file content
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Access variables
    transcript_file = config['DEFAULT']['transcript_file']
    filler_words_file = config['DEFAULT']['filler_words_file']
    transcript_auto_fill = config.getboolean("DEFAULT", "transcript_auto_fill")

    main(transcript_file=transcript_file,
         filler_words_file=filler_words_file,
         transcript_auto_fill=transcript_auto_fill)

