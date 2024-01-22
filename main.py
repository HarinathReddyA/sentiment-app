import streamlit as st
from transformers import pipeline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load the sentiment analysis model from Hugging Face
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_analyzer(text)
    print(result)
    positive_score = result[0]['score']
    label = result[0]['label']
    return positive_score,label

def create_sentiment_gradient_bar(positive_score,colors):
    width = 10
    height = 1

    # Use positive sentiment score to determine the color distribution
    num_color1_pixels = int(width * positive_score)
    num_color2_pixels = width - num_color1_pixels

    gradient = np.concatenate([np.linspace(0, 0.2, num_color1_pixels),
                               np.linspace(1, 0.8, num_color2_pixels)])

    gradient = np.vstack((gradient, gradient))

    fig, ax = plt.subplots(figsize=(width, height))
    cmap = LinearSegmentedColormap.from_list('gradient', colors, N=256)
    ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, width, 0, height])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    st.pyplot(fig)

# Streamlit app
st.title("Sentiment Analysis and Gradient Bar Generator")

# Text input for user
user_input = st.text_input("Enter text for sentiment analysis:")

# Button to trigger sentiment analysis and display the gradient bar
if st.button("Submit"):
    if user_input:
        st.subheader("Text Sentiment Analysis Result:")
        positive_score,label = analyze_sentiment(user_input)
        if label !='NEGATIVE':
            st.write(f"Positive Sentiment Score: {round(positive_score,4)}")
            st.write(f"Negative Sentiment Score: {round(1-positive_score,4)}")
            colors = ['green','red']
        else:
            st.write(f"Negative Sentiment Score: {round(positive_score,4)}")
            st.write(f"Positive Sentiment Score: {round(1-positive_score,4)}")
            colors = ['red','green']


        st.subheader("Sentiment Bar:")
        create_sentiment_gradient_bar(positive_score,colors=colors)
    else:
        st.warning("Please enter text for sentiment analysis.")
