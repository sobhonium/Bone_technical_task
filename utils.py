"""
Author: Sobhan
Last update: 28-05-2025
Project: Bones Ltd. job technical challenge. 
File: utils.py
Description and Usage:
   Utility functions that should be accompanied with main files.
"""

from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
import text2emotion as te
import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tags import TagOutput
from langchain.chains import create_tagging_chain_pydantic

def setup_llm():
    load_dotenv()
    api_key = os.getenv("Groq_API_KEY")
    if not api_key:
        raise ValueError("Missing Groq_API_KEY in environment variables.")
    return ChatGroq(model="llama3-8b-8192", groq_api_key=api_key)

def read_fillers_from_file(file_path):
    '''Reads and returns all fillers in the given file at file_path.'''
    try:
        fillers = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    fillers.append(line)
        return fillers
    except FileNotFoundError:
        print(f"The file '{file_path}' was not found.")
        return []

def gen_auto_sample_conversation():
    '''Generates a sample conversation using Groq's LLAMA model.'''
    try:
        llm = setup_llm()
        messages = [
            SystemMessage(content="""
                you are a helpful assistant giving sample conversations with different sentiments (positive, neutral and negative) on daily conversation topics.

                ## sample style :
                - The following shows a sample conversation:
                   Speaker A: Hey, did you catch the game last night?
                   Speaker B: Yeah, I did! It was, like, really intense toward the end.
                   ...

                ## Rules:
                - include at least 3 filler words in each speaker's turn.
                - speakers must be Speaker A and Speaker B
                - avoid introductory text in response
            """),
            HumanMessage(content="Give a daily conversation between two people with at least 16 alternative talks. Please give different sentiments.")
        ]
        response = llm(messages)
        if not hasattr(response, 'content') or not response.content.strip():
            raise ValueError("Received an empty or invalid response from the language model.")
        return response.content.replace('\n\n', '\n')
    except Exception as e:
        return f"Error generating conversation: {e}"

def load_conversation(auto_fill=False, file_path='transcript.txt'):
    '''Loads or generates a conversation.'''
    if auto_fill:
        conversation = gen_auto_sample_conversation()
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(conversation)
        return conversation
    else:
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")

def render_and_visualize(inputs):
    '''Visualizes conversation and analysis results using Streamlit.'''
    st.markdown("""
        <style>
        .chat-container { display: flex; flex-direction: column; }
        .chat-bubble { max-width: 100%; padding: 5px; margin: 2px; border-radius: 15px; font-size: 16px; }
        .left { background-color: #f1f0f0; text-align: left; margin-right: 10%; align-self: flex-start; }
        .right { background-color: #f4C2f2; text-align: right; margin-left: 10%; align-self: flex-end; }
        .meta { font-size: 12px; margin-top: 5px; opacity: 0.8; }
        </style>
    """, unsafe_allow_html=True)

    st.title("\U0001F4AC Conversation Viewer and Analysis")
    st.subheader("Conversation Display")
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    show_opt = st.selectbox("Select Display Option", ["None", "Show Conversation"])

    if show_opt == "Show Conversation":
        for entry in inputs:
            speaker = entry['speaker']
            message = entry['message'].strip()
            sentiment = entry.get('sentiment', 'NEUTRAL')
            score = entry.get('score', 0.0)
            filler_words_ratio = entry.get('filler_words_ratio', 0.0)

            alignment = "left" if speaker == "Speaker A" else "right"
            sentiment_color = {"POSITIVE": "limegreen", "NEGATIVE": "crimson", "NEUTRAL": "gray"}.get(sentiment.upper(), "black")

            bubble_html = f"""
                <div class="chat-bubble {alignment}">
                    <strong>{speaker}:</strong><br>{message}
                    <div class="meta">
                        <span style='color:{sentiment_color}'>Sentiment: {sentiment} ({score:.2f})</span><br>
                        <span style='color:blue'>Filler Ratio: {filler_words_ratio:.2f}</span>
                    </div>
                </div>
            """
            st.markdown(bubble_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    df = pd.DataFrame(inputs)
    df['sentiment_score'] = df.apply(lambda row: row['score'] if row['sentiment'] == 'POSITIVE' else -row['score'], axis=1)

    st.subheader("Sentiment Analysis")
    options = st.selectbox("Select Analysis to Show", [
        "None",
        "Show Data Table",
        "Total Sentiment Score per Speaker",
        "Count of Positive vs Negative",
        "Sentiment Progression Over Time",
        "Average Filler Word Ratio",
        "Message Length vs. Sentiment Score",
        "Speaker Turn-Taking Pattern",
        "Sentiment Score Distribution",
        "Average Sentiment per Turn",
        "Emotion Breakdown per Speaker",
        "Conversation Taggings (takes time)"
    ])

    if options == "Show Data Table":
        st.dataframe(df)
    elif options == "Total Sentiment Score per Speaker":
        grouped = df.groupby('speaker')['sentiment_score'].sum().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(data=grouped, x='speaker', y='sentiment_score', palette='coolwarm', ax=ax)
        ax.axhline(0, color='black', linestyle='--')
        ax.set_ylabel("Total Sentiment Score")
        st.pyplot(fig)
    elif options == "Count of Positive vs Negative":
        count_df = df.groupby(['speaker', 'sentiment']).size().unstack(fill_value=0)
        fig, ax = plt.subplots()
        count_df.plot(kind='bar', stacked=True, ax=ax, colormap='coolwarm')
        ax.set_ylabel("Message Count")
        st.pyplot(fig)
    elif options == "Sentiment Progression Over Time":
        df['message_index'] = range(len(df))
        fig, ax = plt.subplots()
        sns.lineplot(data=df, x='message_index', y='sentiment_score', hue='speaker', marker="o", ax=ax)
        ax.set_xlabel("Message Index")
        ax.set_ylabel("Sentiment Score")
        st.pyplot(fig)
    elif options == "Average Filler Word Ratio":
        filler_avg = df.groupby('speaker')['filler_words_ratio'].mean().reset_index()
        fig, ax = plt.subplots()
        sns.barplot(data=filler_avg, x='speaker', y='filler_words_ratio', palette='Blues', ax=ax)
        ax.set_ylabel("Avg. Filler Word Ratio")
        st.pyplot(fig)
    elif options == "Message Length vs. Sentiment Score":
        df['message_length'] = df['message'].apply(lambda x: len(x.split()))
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='message_length', y='sentiment_score', hue='speaker', ax=ax)
        ax.set_xlabel("Message Length (words)")
        ax.set_ylabel("Sentiment Score")
        st.pyplot(fig)
    elif options == "Speaker Turn-Taking Pattern":
        df['turn'] = (df['speaker'] != df['speaker'].shift()).cumsum()
        turn_counts = df.groupby(['turn', 'speaker']).size().reset_index(name='count')
        fig, ax = plt.subplots()
        sns.barplot(data=turn_counts, x='turn', y='count', hue='speaker', dodge=False, ax=ax)
        ax.set_xlabel("Turn Index")
        ax.set_ylabel("Messages in Turn")
        st.pyplot(fig)
    elif options == "Sentiment Score Distribution":
        fig, ax = plt.subplots()
        sns.histplot(data=df, x='sentiment_score', hue='speaker', kde=True, bins=200, ax=ax)
        ax.set_title("Distribution of Sentiment Scores")
        st.pyplot(fig)
    elif options == "Average Sentiment per Turn":
        df['turn'] = (df['speaker'] != df['speaker'].shift()).cumsum()
        turn_sentiment = df.groupby('turn').agg({'speaker': 'first', 'sentiment_score': 'mean'}).reset_index()
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=turn_sentiment, x='turn', y='sentiment_score', hue='speaker', dodge=False, ax=ax)
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel("Turn Index")
        ax.set_ylabel("Avg. Sentiment Score")
        ax.set_title("Average Sentiment per Turn")
        st.pyplot(fig)
    elif options == "Emotion Breakdown per Speaker":
        # uses text2emotion
        df_emotions = df.copy()
        df_emotions['emotions'] = df_emotions['message'].apply(te.get_emotion)
        emotion_df = df_emotions['emotions'].apply(pd.Series)
        df_emotions = pd.concat([df_emotions, emotion_df], axis=1)
        emotion_cols = ['Happy', 'Angry', 'Surprise', 'Sad', 'Fear']
        emotion_summary = df_emotions.groupby('speaker')[emotion_cols].mean().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        emotion_summary.set_index('speaker').T.plot(kind='bar', ax=ax)
        ax.set_ylabel("Average Emotion Score")
        ax.set_title("Emotion Breakdown by Speaker")
        ax.legend(title='Speaker')
        st.pyplot(fig)
    elif options == "Conversation Taggings (takes time)":
        # an ability that overcome the limitation of text2emotion
        # since it can allow you specify the tags and sentimisent
        # for yourself (customized) 
        full_text = ' '.join(df['message'].astype(str))
        llm = setup_llm()
        tagging_chain = create_tagging_chain_pydantic(TagOutput, llm)
        run_properly = False
        # Since llms are not deterministic (and not giving)
        # what we want at once, it's inteded here to rerunn
        # until we are happy with the output of the llm (Groq here)
        while not run_properly:
            try:
                result = tagging_chain.run(full_text)
                st.write(result)
                run_properly = True
            except:
                print('Retrying tagging...')
        print('Done!')
