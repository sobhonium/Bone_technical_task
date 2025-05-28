"""
Author: Sobhan
Last upate: 28-05-2025
Project: Bones Ltd. job technical challenge.
File: app.py
Description and Usage:
    This file includes the main() function and is the entry point
    for running the Streamlit app.
    Suggested run command: > streamlit run app.py
"""

import configparser
import streamlit as st

from utils import read_fillers_from_file, load_conversation, render_and_visualize
from analysis import compute_sentiment


def main(
    transcript_file: str = 'transcript.txt',
    filler_words_file: str = 'filler_words.txt',
    transcript_auto_fill: bool = False,
) -> None:
    """
    Main entry function to load conversation data, compute sentiment,
    and render the Streamlit interface.
    Uses st.session_state to avoid redundant reloads and computations.
    """

    # the 'if' condition avoids loading conversations more than once.
    # So, the strategy is to load and compute sentiments once
    # but show and play with checkbox without reloading 
    # and computing everything. By default, streamlit reloads
    # everything on the file, but this session_state will put a stop
    # to it.
    if "results" not in st.session_state:
        # loading all fillers
        all_filler_words = read_fillers_from_file(filler_words_file)

        # loading conversation 
        conversation = load_conversation(
            auto_fill=transcript_auto_fill,
            file_path=transcript_file,
        )

        # compute sentiments
        results = compute_sentiment(conversation, all_filler_words)

        # st.session_state.computed_already = True
        st.session_state.results = results

        # display with conversation streamlit and plot graphs
        render_and_visualize(inputs=results)

    else:
        results = st.session_state.results
        render_and_visualize(inputs=results)


if __name__ == "__main__":
    
    # Load config file content
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Access variables
    transcript_file = config['DEFAULT'].get('transcript_file', 'transcript.txt')
    filler_words_file = config['DEFAULT'].get('filler_words_file', 'filler_words.txt')
    transcript_auto_fill = config['DEFAULT'].getboolean('transcript_auto_fill', False)

    main(
        transcript_file=transcript_file,
        filler_words_file=filler_words_file,
        transcript_auto_fill=transcript_auto_fill,
    )
