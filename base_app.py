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

# Vectorizer
tweets_vectorizer = open("resources/cv_vect_2.pkl", "rb")
# loading your vectorizer from the pkl file
tweet_cv = joblib.load(tweets_vectorizer)

# Load your raw data
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
    st.title("Tweet Sentiment Classifier")
    st.subheader("Climate change tweet classification")

    app_background_image = """
    <style>
        body {
            background-image: 
                url("https://images.pexels.com/photos/158827/field-corn-air-frisch-158827.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
            background-size: cover;
            background-repeat: no-repeat;
        }
    </style>
    """

    st.markdown(
        app_background_image,
        unsafe_allow_html=True
    )


    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction","Problem Statement", "Dataset", ]
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

    if selection == "Problem Statement":
        st.write("""
        ![Climate Change Image](https://github.com/RimanaSifiso/public-data-for-ml-tutorials/blob/main/edsa/classification_exam/bg-small.jpg?raw=true)

        ## Defining Cimate Change
        According to [United Nations](https://www.un.org/en/climatechange/what-is-climate-change), 
        climate change is a long-term shift in temperatures and weather patterns, and may be of natural 
        cause or man-made.
        
        ## Why does this matter?


        Recently, companies have been transforming their production methods in an effort to produce environmentally 
        friendly products. According to [National News Article](https://www.thenationalnews.com/business/technology/2022/06/23/25-of-top-tech-companies-set-to-achieve-carbon-neutrality-by-2030-report-says/), 
        `25%` of top tech companies are set to achieve carbon neutrality by 2030.

        Interestingly, people tend to buy products from companies that align with their values and beliefs. 
        According to [Giusy Bounfantino's article](https://consumergoods.com/new-research-shows-consumers-more-interested-brands-values-ever#:~:text=Shoppers%20want%20to%20buy%20from%20brands%20aligned%20with%20their%20values&text=The%20new%20research%20tells%20us,over%20a%20conflict%20in%20values.) 
        on Consumer Goods Technology, `82%` of shoppers want a consumer brand’s values to align with their own.

        Since companies are now producing environmentally friendly products, they want to know what the general public 
        thinks about climate change, as that is likey to determine how their products will be received by the public. 
        A person who believes climate change is a real threat is more likely to buy environmentally friendly products, 
        or buy from a company that has the same belief.
        """)

    # Building out the predication page
    if selection == "Prediction":
        st.info("""
        Sometimes it’s hard to tell whether a tweet is pro, anti, neutral, or news about 
        climate change, or it could be that the tweet is long to read to classify it manually. 
        Save the hard work for us, enter the tweet here and we’ll tell you whether it is pro, 
        anti, neutral, or news about climate change.
        """)
        st.write("### Classify tweets the _classy_ way")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter tweet to classify", "Type Here...")

    

        if st.button("Classify"):
            # Transforming user input with vectorizer
            tweet_text = nlp_preprocess(tweet_text)
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(
                open(os.path.join("resources/lr_model.pkl"), "rb"))
            prediction = predictor.predict(vect_text)

            output = ''

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
