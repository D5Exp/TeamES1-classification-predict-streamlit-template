"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib
import os

# Data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


raw = pd.read_csv("resources/train.csv")

# ---------------- FUNCTION FOR PRE-PROCESSING TEXT ------------


def nlp_preprocess(tweet_text):
    """
    Performs the following Natural Language Processing steps on the incoming tweet text:
    1. Text cleaning, noise removal (web-urls, punctuations, etc.)
    2. Converting text to lowercase
    3. Stopwords removal
    4. Tokenization
    5. Stemming
    6. Lemmatization
    
    :param tweet_text: string
    :return: string
    """

    import nltk
    from string import punctuation
    from nltk.tokenize import TreebankWordTokenizer
    from nltk.corpus import stopwords
    from nltk import SnowballStemmer
    from nltk.stem import WordNetLemmatizer

    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    # Removing URLs
    url_pattern = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
    tweet_text = tweet_text.replace(url_pattern, '',)

    # Removing RTs
    tweet_text = tweet_text.replace('RT', '')

    # Removing hashtags
    tweet_text = tweet_text.replace(r'#([A-Za-z0-9_]+)', '')

    # Removing Twitter handles (i.e. @user)
    tweet_text = tweet_text.replace(r'@([A-Za-z0-9_]+)', '')

    # Removing punctuations
    tweet_text = ''.join([l for l in tweet_text if l not in punctuation])

    # CONVERTING TO LOWERCASE
    # stripping off potential trailing spaces
    tweet_text = tweet_text.lower().strip()

    # REMOVE UNKKNOWN CHARS AND NUMBERS
    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
               "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " "]
    tweet_text = ''.join([l for l in tweet_text if l in letters])

    # LEMMATIZATION
    lemmatizer = WordNetLemmatizer()

    tweet_text = ''.join([lemmatizer.lemmatize(word) for word in tweet_text])

    return tweet_text


# The main function where we will build the actual app


def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.set_page_config(page_title="Tweet Sentiment app")
    st.title("Tweet Sentiment Classifier")
    st.subheader("Climate change tweet classification")


    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Home","New Predictions", "Dataset",]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Dataset":
        st.write("# Nature of the dataset")
        # You can read a markdown file from supporting resources folder
        st.markdown("""
        ![image-3.png](https://raw.githubusercontent.com/RimanaSifiso/public-data-for-ml-tutorials/main/edsa/classification_exam/icons8-twitter-48.png)

        Today’s voices are in social media. People express their values or beliefs on particular topics or 
        areas of life mostly on social media platforms. The dataset is a collection of tweets from 2015 to 
        2018 about climate change. 

        
        """)

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'):  # data is hidden if box is unchecked
            # will write the df to the page
            st.write(raw[['sentiment', 'message']])

    if selection == "Home":
        st.write(
            """
            ## Do you have to care about what your customers say?

            Studies show that `82%` of customers want a brand's values that aligns align with their own.

            When it comes to product production and the environment, looking at whether people care about climate change or not
            is as crucially important as your product success. According to the data used in our study, almost more than half of the population perceive climate change as a human impact.
            """
        )


        pie_data = raw['sentiment'].map({2: 'News', 1: 'Pro', 0: 'Neutral', -1: 'Anti'}
                                        ).value_counts()

        fig, ax = plt.subplots()

        ax.pie(x=pie_data, autopct='%.2f%%', labels=pie_data.index)

        plt.title("Tweets Sentiments by Count")
        plt.ylabel("")

        st.pyplot(fig)

        st.write(
            """
            ## The Problem we solved

            To analyze customr sentiments about cilmate change, you'll need large amounts of data, and analyzing large amounts of data is time and cost consuming, our app allows use you to skip that tedious task.

            Simply collect customer data and use "New Predictions" analyze the results in your dataset, with just a click!
            """
        )


    # Building out the predication page
    if selection == "New Predictions":
        st.info("""
        Sometimes it’s hard to tell whether a tweet is pro, anti, neutral, or news about 
        climate change, or it could be that the tweet is long to read to classify it manually. 
        Save the hard work for us, enter the tweet here and we’ll tell you whether it is pro, 
        anti, neutral, or news about climate change.
        """)
        st.write("### Classify tweets the _classy_ way")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter tweet to classify", "Type Here...")

        predictor = None 
        tweets_vectorizer = None

        # allow the use to choose the model for prediction
        model = st.radio(
            "Choose a model to use for predictions",
            ("Logistic Regression", "Desision Tree")
        )

        if model == "Logistic Regression":
            tweets_vectorizer = open("resources/cv_vect_2.pkl", "rb")
            tweet_cv = joblib.load(tweets_vectorizer)
            predictor = joblib.load(
                open(os.path.join("resources/lr_model.pkl"), "rb"))
        else:
            tweets_vectorizer = open("resources/cv_vect.pkl", "rb")
            tweet_cv = joblib.load(tweets_vectorizer)
            predictor = joblib.load(
                open(os.path.join("resources/dtc_model.pkl"), "rb"))

        if st.button("Classify"):
            # Transforming user input with vectorizer
            tweet_text = nlp_preprocess(tweet_text)
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            
            prediction = predictor.predict(vect_text)

            output = ''

            # generilize output to be more user friendly
            if prediction == 2:
                output = "Climate change factual news"
            elif prediction == 1:
                output = "Pro: the tweet is positive about human made climate change"
            elif prediction == -1:
                output = "Anti: the tweet is negative about human made climate change"
            elif prediction == 0:
                output = "Neutral: the tweet is neutral about human made climate change"
            else:
                st.error('The tweet could not be classified.')
            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success(output)


# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()
