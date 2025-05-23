# Repo Description
Building a Python app that analyzes a short dialogue transcript for sentiment and filler-word usage, then displaying the results in an interactive dashboard using Streamlit.

# Setup/run Instructions
1. Create the environment

You can create a new Conda environment and install packages from requirements.txt using:

 <pre>conda create --name myenv python=3.10 
conda activate myenv </pre>

2. Install packages from requirements.txt

 <pre>pip install -r requirements.txt</pre>


3. Specify the configurations. This includes the ```transcript_file``` 
 and ```filler_words_file``` file paths that are needed to be specified. In addition to these, if you want to use the content of transcript that is already filled with a conversation, leave ```transcript_auto_fill=flase```, but if you want it to be automatically filled set is as  ```transcript_auto_fill=true```. This will use an LLM with prompt (already well-engineered) to give a sample conversation. The defualt values are:
<pre>
[DEFAULT]
transcript_file = transcript.txt
filler_words_file = filler_words.txt
transcript_auto_fill = false
</pre>

If you are preferring `transcript_auto_fill = true`, you need to set `Groq_API_KEY=<API-Key>` in `.env` file. This setup is meant to use inference platform Groq (fortunetly, fully opensource) with deafult model=`llama3-8b-8192` to fill the `transcript_file` file.

The `HumanMessage` and `SystemMessage` for the prompt is already put there to generate a reliable conversation.

4. To run the project and render the results:

<pre>streamlit run app.py</pre>


# 🔹  Notes
- If you want to try Spacy lib for filler word analysis, first you need to download ```en_core_web_sm``` by the following terminal command (Specified in the code).

 <pre> python -m spacy download en_core_web_sm  </pre>
then you need to comment(or uncomment) the relavant code snippet in ```analysis.py``` file.

- for Emotion analysis (a plus to sentiment analysis), you can install other tools like



<pre>pip install text2emotion
pip install nltk   
python -m nltk.downloader all</pre>



# Descriptions of each metric

### Show Total sentiment score 
It shows the total sentiments per speaker by summing over all scores throughout the conversation. This will help to show how positive or negagive a person is during the conversation.

### Sentiment Score Distribution
 Distribution of sentiment polarity per speaker — are they generally positive, neutral, or extreme? This measurement is showing the score on each part (not the overall summing score).

### Show Count of positive vs negative
Regardless of the score,  this measurement counts the number of positive vs negative messages from each person.   

### Sentiment Progression Over Time
Helps find the sentiment progression over time. This might be useful when we want to see how conversation has made a speaker change his tone, etc.

###  Average Filler Word Ratio
I think it can show speaking clarity or fluency of a Speaker. The sheer number of filler words from a person cannot be representing  clarity.

### Message Length vs. Sentiment Score
It explores correlation between verbosity and sentiment. Can we see in the conversation that the more words or explanation someone is using has a connection with a sentiment (just a hypothesis)?


### Average Sentiment per Turn
I don't know what it means by Turn in conversation, so, I refer to ChatGPT for that. So, What Is a Turn in conversation? (asked ChatGPT) 
 - reply by ChatGPT: A turn is a group of consecutive messages from the same speaker without interruption by the other. 

 So, I think the average sentiment per trun is useful for measuring how a speaker's sentiment evolves over their stretches of speech.

### Speaker Turn-Taking Pattern
What it shows: Whether the conversation is balanced or dominated by one speaker. If Speaker 1, leaves two messaging without allowing the Speaker 2 to say anything in between, it means that the conversation at that round has inbalance. The prefect balance is when we have 1-1 chats and no one talks more in one turn.




### Emotion Breakdown per Speake
Computes emotion scores for each message. For this, `text2emotion` is used which automatically calculates the following emotions in text:
<pre>emotion_cols = ['Happy', 'Angry', 'Surprise', 'Sad', 'Fear']</pre>

# Potential Issues

The `text2emotion` package relies on an older version of the `emoji` library, where `emoji.UNICODE_EMOJI` or `emoji.EMOJI_DATA` was available. In more recent versions (from `emoji>=2.0.0`), these internal data structures were moved or removed. If you run into errors, you need to replace   `emoji.UNICODE_EMOJI` with `emoji.EMOJI_DATA` in 
```..../miniconda3/envs/<env-name>/lib/python3.10/site-packages/text2emotion/__init__.py``` file in your file system.

# Results (screenshots)

See the pdf ourput files uploaded as `output1.pdf` and `output2.pdf`.

# In one extra hour I would add
###### (it might however be more than an hour but I'm just keeping it here)
- add `transcript.txt` and `filler_words.txt` into a folder called `/data`  as it's a more common way of using input files.

- When generating auto conversations, `langchain` offers a json parser for generating a standard output. I would definely go for that since, in some cases (eventhough the prompt is set correctly), some unwanted words might be presented by LLM in the begining of a text which is not ideal. Using that parser can grauantee such output structure for conversations. see [1].

- As of now, the unit test functions are tightly coupled with input files which is not the best programming practice. It's best if the testing function being decoupled. By that, I mean the functions should be well established and work for all inputs and check things based on all inputs. But, now they just consider hardcoded files like `transcripts.txt` or `filler_words.txt`.

- In `gen_auto_sample_conversation()` function `SystemMessage` and `HumnanMessage` for LLM can be put in another file to read them (cleaner code practice I believe). Or in `render_and_visualize()` function, `css` codes can be in separate files.

- As might be obvious, if we have extra resources, they can be spent on researching what meaningful information can be captured from conversations, as is common practice in conversation analysis. I crudely asked ChatGPT about such things, but it seems like they require additional third-party tools (LDA, BERTopic, Top2Vec, AllenNLP, TextBlob) and more research (assuming the answer isn't hallucinated).

# Resource
[1] https://nanonets.com/blog/langchain/?utm_source=chatgpt.com#module-ii-retrieval
