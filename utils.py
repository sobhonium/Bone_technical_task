"""
Author: Sobhan
Date: 23-05-2025
Project: Bones Ltd. job technical challenge. 
File: utils.py
Description and Usage:
   unitilty function that should be accompanied with main files.
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


def read_fillers_from_file(file_path):
    '''reads and returns all fillers in the given filat at file_path'''
    try:
        fillers = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()  # Remove leading/trailing whitespace
                if line:  # Avoid empty lines
                    fillers.append(line)

        return fillers

    except FileNotFoundError:
        print(f"The file '{file_path}' was not found.")
        return []


def gen_auto_sample_conversation():
    '''It generates an auto conversation using Groq platform and LLAMA model.
    '''
    try:

        load_dotenv()

        groq_api_key = os.getenv("Groq_API_KEY")

        if not groq_api_key or groq_api_key.strip() == "":
            raise ValueError(
                "Groq_API_KEY is missing or empty in the .env variables.")

        # Initialize the LLM with your model and API key
        llm = ChatGroq(
            model="llama3-8b-8192",
            groq_api_key=groq_api_key
        )

        # Provide system and human messages
        messages = [
            SystemMessage(content="""
         you are a helpful assistant giving sample conversations with different sentiments (positive, neutual and negative) on daily conversation topics.
         
         
         ## sample style :
            - The following shows a sample conversation:
               Speaker A: Hey, did you catch the game last night?
               Speaker B: Yeah, I did! It was, like, really intense toward the end.
               Speaker A: I know, right? I thought they were gonna lose for sure.
               Speaker B: Same here. And then, um, that last-minute goal? Unreal.
               Speaker A: Totally. I was actually yelling at the TV.
               Speaker B: Haha, my dog freaked out when I did that. Poor guy.
               Speaker A: So, you think they're gonna make the playoffs?
               Speaker B: I mean, if they keep playing like this... maybe. It’s hard to say.
               Speaker A: Yeah, consistency hasn’t really been their thing this season.
               Speaker B: True. And, you know, the coach keeps switching up the lineup.
               Speaker A: That might be part of the problem. No one gets into a rhythm.
               Speaker B: Exactly. Oh, did you see who they’re playing next week?
               Speaker A: I think it’s Boston. That’s gonna be rough.
               Speaker B: Ugh, yeah. Boston's defense is no joke.
               Speaker A: Well, we’ll see. Stranger things have happened.
               Speaker B: For sure. Fingers crossed!

         ## Rules:
            - you should include at least 3 filler words in each speaker's turn. 
            - The speakers names must be Speaker A and Speaker B
            - The resonse you give should not contain anything but the conversation. I mean avoid the begining text reponse like:
            Here is a conversation between Speaker A and Speaker B with at least 16 alternative talks.
            - avoid the starts like : Here is a conversation between Speaker A and Speaker B just go stright with Speaker A:

         """),

            HumanMessage(content="""Give a daily conversation between two people with at least 16 alternative talks. please give different sentiments (positive, neutual and negative).""")
        ]

        # Get the response
        response = llm(messages)

        # Validate response
        if not hasattr(response, 'content') or not response.content.strip():
            raise ValueError(
                "Error: Received an empty or invalid response from the language model.")

        return response.content.replace('\n\n', '\n')

    except Exception as e:
        # Log the error or return a message indicating failure
        return f"Error generating conversation: {e}"


def load_conversation(auto_fill=False, file_path='transcript.txt'):
    '''returns a conversation whether it's automatic or already in file_path'''

    if auto_fill == True:
        conversation = gen_auto_sample_conversation()

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(conversation)

        return conversation

    else:  # then load the file_path content

        try:
            with open(file_path, 'r') as file:
                conversation = file.read()  # Read the whole file as a single string
            return conversation

        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")


def render_and_visualize(inputs):
    '''Renders the conversation and plots for streamlit to display them'''

    st.markdown(
        """
        <style>
        .chat-container {
            display: flex;
            flex-direction: column;
        }
        .chat-bubble {
            max-width: 100%;
            padding: 5px 5px;
            margin: 2px;
            border-radius: 15px;
            font-size: 16px;
            line-height: 1.4;
            display: inline-block;
            word-wrap: break-word;
        }
        .left {
            background-color: #f1f0f0;
            text-align: left;
            border-top-left-radius: 0;
            margin-right: 10%;
            align-self: flex-start;
        }
        .right {
            background-color: #f4C2f2;
            color: black;
            text-align: right;
            border-top-right-radius: 0;
            margin-left: 10%;
            align-self: flex-end;
        }
        .meta {
            font-size: 12px;
            margin-top: 5px;
            opacity: 0.8;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("💬 Conversation Viewer and Analysis")
    st.subheader("Conversation Display")
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    if st.checkbox("Show Conversation"):
        for entry in inputs:
            speaker = entry['speaker']
            message = entry['message'].strip()
            sentiment = entry.get('sentiment', 'NEUTRAL')
            score = entry.get('score', 0.0)
            filler_words_ratio = entry.get('filler_words_ratio', 0.0)

            alignment = "left" if speaker == "Speaker A" else "right"

            sentiment_color = {
                "POSITIVE": "limegreen",
                "NEGATIVE": "crimson",
                "NEUTRAL": "gray"
            }.get(sentiment.upper(), "black")

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


    # Other optional plots and graphs:  
    # Convert to DataFrame
    df = pd.DataFrame(inputs)

    # Add sentiment score (+/- based on sentiment)
    df['sentiment_score'] = df.apply(
        lambda row: row['score'] if row['sentiment'] == 'POSITIVE' else -row['score'],
        axis=1
    )

    
    st.subheader("Sentiment Analysis")

    # ---- Display raw data ----
    if st.checkbox("Show data table"):
        st.dataframe(df)

    # ---- total sentiments by summing over all scores----
    if st.checkbox("Show Total sentiment score per speaker"):
      grouped = df.groupby('speaker')['sentiment_score'].sum().reset_index()
      fig1, ax1 = plt.subplots()
      sns.barplot(data=grouped, x='speaker', y='sentiment_score',
                  palette='coolwarm', ax=ax1)
      ax1.axhline(0, color='black', linestyle='--')
      ax1.set_ylabel("Total Sentiment Score")


      st.pyplot(fig1)

    # ---- Count of positive vs negative messages ----
    if st.checkbox("Show Count of positive vs negative"):
      count_df = df.groupby(['speaker', 'sentiment']
                           ).size().unstack(fill_value=0)

      fig2, ax2 = plt.subplots()
      count_df.plot(kind='bar', stacked=True, ax=ax2, colormap='coolwarm')
      ax2.set_ylabel("Message Count")

    
      st.pyplot(fig2)

    # ----- Sentiment Progression Over Time ---
    if st.checkbox("Sentiment Progression Over Time"):
        fig3, ax3 = plt.subplots()
        df['message_index'] = range(len(df))
        sns.lineplot(data=df, x='message_index', y='sentiment_score',
                     hue='speaker', marker="o", ax=ax3)
        ax3.set_xlabel("Message Index")
        ax3.set_ylabel("Sentiment Score")
        st.pyplot(fig3)

    # I think it shows speaking clarity or fluency.
    if st.checkbox("Average Filler Word Ratio per Speaker"):
        filler_avg = df.groupby('speaker')[
            'filler_words_ratio'].mean().reset_index()

        fig4, ax4 = plt.subplots()
        sns.barplot(data=filler_avg, x='speaker',
                    y='filler_words_ratio', palette='Blues', ax=ax4)
        ax4.set_ylabel("Avg. Filler Word Ratio")
        st.pyplot(fig4)

    # Explore correlation between verbosity and sentiment.
    if st.checkbox("Message Length vs. Sentiment Score"):
        df['message_length'] = df['message'].apply(lambda x: len(x.split()))

        fig5, ax5 = plt.subplots()
        sns.scatterplot(data=df, x='message_length',
                        y='sentiment_score', hue='speaker', ax=ax5)
        ax5.set_xlabel("Message Length (words)")
        ax5.set_ylabel("Sentiment Score")
        st.pyplot(fig5)

    # What it shows: Whether the conversation is balanced or dominated by one speaker.
    if st.checkbox("Speaker Turn-Taking Pattern"):
        df['turn'] = (df['speaker'] != df['speaker'].shift()).cumsum()
        turn_counts = df.groupby(
            ['turn', 'speaker']).size().reset_index(name='count')

        fig, ax = plt.subplots()
        sns.barplot(data=turn_counts, x='turn', y='count',
                    hue='speaker', dodge=False, ax=ax)
        ax.set_xlabel("Turn Index")
        ax.set_ylabel("Messages in Turn")
        st.pyplot(fig)

    # What it shows: Distribution of sentiment polarity per speaker — are they generally positive, neutral, or extreme?
    if st.checkbox("Sentiment Score Distribution"):
        fig, ax = plt.subplots()
        sns.histplot(data=df, x='sentiment_score',
                     hue='speaker', kde=True, bins=200, ax=ax)
        ax.set_title("Distribution of Sentiment Scores")
        st.pyplot(fig)
    
    # average sentiment per turn
    if st.checkbox("Average Sentiment per Turn"):
      
      df['turn'] = (df['speaker'] != df['speaker'].shift()).cumsum()

      turn_sentiment = df.groupby('turn').agg({
         'speaker': 'first',
         'sentiment_score': 'mean'
      }).reset_index()
      fig, ax = plt.subplots(figsize=(10, 4))
      sns.barplot(data=turn_sentiment, x='turn', y='sentiment_score', hue='speaker', dodge=False, ax=ax)
      ax.axhline(0, color='gray', linestyle='--')
      ax.set_xlabel("Turn Index")
      ax.set_ylabel("Avg. Sentiment Score")
      ax.set_title("Average Sentiment per Turn")
      st.pyplot(fig)

    # Compute emotion scores for each message
    df_emotions = df.copy()
    df_emotions['emotions'] = df_emotions['message'].apply(te.get_emotion)

    # Split emotion dict into separate columns
    emotion_df = df_emotions['emotions'].apply(pd.Series)
    df_emotions = pd.concat([df_emotions, emotion_df], axis=1)

    if st.checkbox("Emotion Breakdown per Speaker"):
        emotion_cols = ['Happy', 'Angry', 'Surprise', 'Sad', 'Fear']
        emotion_summary = df_emotions.groupby(
            'speaker')[emotion_cols].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        emotion_summary.set_index('speaker').T.plot(kind='bar', ax=ax)
        ax.set_ylabel("Average Emotion Score")
        ax.set_title("Emotion Breakdown by Speaker")
        ax.legend(title='Speaker')
        st.pyplot(fig)
