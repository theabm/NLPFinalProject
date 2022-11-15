># Topic Modeling and Sentiment Analysis on Amazon book dataset

The main aim of this project is to implement tools based on NLP techniques to be used for the following tasks:

1. Help publishers and authors understand the topics of books being sold on Amazon to have a better idea of the current interests and overall market situation.
2. Classify user reviews in order to incorporate this knowledge in a collaborative based filtering technique where similar tastes between users are used to recommend new items.   

To address these problems, we perform topic modeling and sentiment analysis on a corpus of book summaries and corresponding amazon reviews.

After studying the most frequent words in the corpus of book descriptions, we extract the K main topics that characterize books by means of LDA, along with the 10 most relevant words for each topic. Consequently, we try to interpret the theme of each topic and analyze their association to the categories provided in the dataset. Lastly, by using the topic distribution of each document, we study how the popularity of each theme changes over time.
    
For sentiment analysis, we initally use pretrained multilingual BERT to predict the score of the reviews from 1 to 5. Then we use RoBERTa to classify the reviews as positive and negative and compare them to grouped ground truth labels. Lastly, we fine tune RoBERTa using HuggingFace's Trainer environment to improve the model on our own data. We compare this model to a baseline dummy classifier which predicts the most frequent class. 

># Running the code
All code is contained and described in the python notebook. It is recommended to use GPU's to accelerate the fine tuning for sentiment analysis. 

># The report
The report describes the main results and explains the procedure.
