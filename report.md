---
title: Topic Modeling and Sentiment Analysis for Amazon Reviews
author: Andres Bermeo Marinelli
date: November 10, 2022
---

# Index
- [Problem Statement](#problem-statement)
- [Data Description](#data-description)
- [Assessment & Performance Indexes](#assessment-and-performance-indexes)
- [Topic Modeling](#topic-modeling)
- [Sentiment Analysis](#sentiment-analysis)
- [Results and Discussion][#results-and-discussion]


---
> # Problem Statement

The main aim of this project is to implement tools based on NLP techniques to be used for the following tasks:

1. Help publishers and authors understand the topics of books being sold on Amazon to have a better idea of the current interests and overall market situation.
2. Classify user reviews in order to incorporate this knowledge in a collaborative based filtering technique where similar tastes between users are used to recommend new items.   

To address these problems, we perform topic modeling and sentiment analysis on a corpus of book summaries and corresponding amazon reviews.

After studying the most frequent words in the corpus of book descriptions, we extract the K main topics that characterize books by means of LDA, along with the 10 most relevant words for each topic. Consequently, we try to interpret the theme of each topic and analyze their association to the categories provided in the dataset. Lastly, by using the topic distribution of each document, we study how the popularity of each theme changes over time.
    
For sentiment analysis, we initally use pretrained multilingual BERT to predict the score of the reviews from 1 to 5. Then we use RoBERTa to classify the reviews as positive and negative and compare them to grouped ground truth labels. Lastly, we fine tune RoBERTa using HuggingFace's Trainer environment to improve the model on our own data. We compare this model to a baseline dummy classifier which predicts the most frequent class. 

---
> # Data Description

The dataset contains two separate files with data on books and their reviews. 

The first file contains details about $\sim 212,000$ books. In particular, it contains information such as title, author, published date, description, and category. We immediately realized that around $68,000$ ($30\%$) items had a missing description, so they were eliminated since our main focus is to extract topics from the summaries of books. Furthermore, we realized later, not all the books were in English. Since LDA implicitly assumes the same language for document creation, we removed documents not in english after applying language recognition tools to the summaries. This left a total of $\sim 142,000$ items. In this subset, we study the frequency of the categories and find that the three most popular are fiction, history, and religion.

<img src="./Images/categories.png" alt="categories" width="450" height = "300" />




The second file contains information about $\sim 3$ million reviews on the books contained in the first file. In particular, we have info such as book id (being reviewed), title of the book, id of user reviewing, summary of the review, text of the review, and finally a score from 1 to 5. The scores are severly imbalanced with 5 star reviews being the most predominant class.

<img src="./Images/scores.png" alt="categories" width="450" height = "300" />


---
> # Assessment and Performance Indexes
With regard to parameter tuning, for topic modeling, LDA requires as input the number of topics $K$. To choose this value, we vary $K$ in $[2,...,20]$ and use two metrics: UMass and CV score. The optimal number of topics is reached when these two scores are locally maximized. However, we also keep in mind the principle of Occam's razor which prefers simpler, more explainable models, sometimes at the expense of a lower metric/score. 

On the other hand, for sentiment analysis, we use accuracy, precision, recall, and f1 score. In particular, we used sklearn's classification report to quicly obtain a per class score of the last three metrics as well as an overall accuracy of the model. In the end, when we compare the baseline with fine tuned RoBERTa, we use balanced accuracy, micro, macro, and weeighted f1 score. This is done to properly study the effect of class imbalance.

---
> # Topic Modeling
After having eliminated items with empty book descriptions and non english text, we are left with $\sim 142000$ elements. At this point, to obtain an even more descriptive summary of the book, we concatenate the title of the book at the end of the summary.

The we aggressively preprocess the text to reduce variance to a minimum and capture the most meaning with the word descriptors after running LDA. In particular, we normalize the text and remove all symbols and punctuations (except for hyphens and apostrophes), we tokenize using spacy, we keep the lemma of each word, we remove stop words contained in the default list of spacy, removing also words such as "book", "author", "write", "story". Finally, we find the joint collocations with PMI $\ge$ 1.0 and represent them as bigrams. This final bound was chosen to limit the computational power and keep only the most significative words. 

We construct the wordcloud from the preprocessed description to study the most frequent terms in order to have some prior ideas of some important topics/themes in the corpus. 

<img src="./Images/wordcloud.png" alt="wordcloudlda" width="650" height = "550" />

Some of the most frequent words are "work", "world", "life", "history", "time", "love", "family", are very suggestive of the topics present in the corpus. These are powerful indicators of possible topics that we may find in the corpus. Also, we can see that they can be linked in some way to the most popular categories shown before. 

After having prepared the corpus for LDA, we must determine the number of topics to extract. To achieve this, we loop over $K = [2,...,20]$ and train LDA on the first $10,0000$ documents, and compute UMass and CV scores on the next $40,000$. We set $\alpha = 0.01$ and $\eta = $"auto". The former tells us to expect around one topic per document while the latter is related to the specificy of words per topic and enabled LDA to learn a prior from the corpus. We obtain the following curves for varying $K$:

<img src="./Images/umasscvscore.png" alt="categories" width="650" height = "400"/>



As we can see, we obtain that the the overall maximum is at $K=10$ topics for both UMass and CV score. 

Finally, we can extract the top 10 descriptors for each topics and examine the results.

|            |**Word 1**|**Word 2**|**Word 3**|**Word 4** |**Word 5**|**Word 6**|**Word 7**|**Word 8**|**Word 9**   |**Word 10**|
|----------- |----------|----------|----------|-----------|----------|----------|----------|----------|-------------|----------|
|**Topic 1** |recipe    |food      |guide     |cookbook   |cook      |garden    |plant     |dish      |cooking      |eat       |
|**Topic 2** |child     |life      |god       |love       |parent    |help      |dog       |little    |animal       |baby      |
|**Topic 3** |social    |theory    |political |study      |history   |science   |economic  |culture   |human        |american  |
|**Topic 4** |music     |art       |film      |work       |artist    |original  |include   |guide     |song         |history   |
|**Topic 5** |bible     |god       |church    |christian  |jesus     |testament |biblical  |spiritual |study        |christ    |
|**Topic 6** |student   |guide     |edition   |provide    |business  |design    |language  |include   |information  |system    |
|**Topic 7** |love      |novel     |man       |life       |find      |woman     |murder    |family    |young        |new       |
|**Topic 8** |game      |baseball  |sport     |guide      |player    |bird      |quilt     |include   |color        |history   |
|**Topic 9** |poem      |poetry    |poet      |litarature |work      |american  |history   |publish   |english      |essay     |
|**Topic 10**|war       |history   |american  |world      |life      |man       |america   |battle    |year         |new       |


As we can see, the topics are quite easy to interpret:
- topic 1: **cooking**
- topic 2: **family**.
- topic 3: **socio economic science**.
- topic 4: **fine arts**.
- topic 5: **religion**.
- topic 6: **student help**.
- topic 7: **fictional romance**.
- topic 8: **sport & outdoors**.
- topic 9: **poetry & literature**.
- topic 10: **history**.

We notice that some of these themes reconnect to the top 10 categories: **History**, **Religion**, **Juvenile Non-Fiction (student help)**, **social science & business/economics (socio economic science)**, **fictional romance** which could be relabeled to **juvenile fiction**.

Furthermore, we observe that some themes are new and others from the most popular categories are missing. For example, there is no presence of **cooking**, **sport & outdoors**, and **poetry & literature** in the most popular categories. On the other hand, our topics seem to be lacking the presence of topics related to **science** and **computers**. It could be possible that these were incorporated into topic 3. 

However, these are surprising discoveries that give more insight into the corpus which goes a level further than if we were to look at just the categories; and in the case that this information is missing, it is a great source of information.

Finally, we analyze how the topics change throughout the years.

In particular, we will analyze the last 12 years (for graphical and practical reasons)

We associate to each document the most relevant document and plot how many items belong to each topic per year. Below we can see the results:


<img src="./Images/topicsbyyear.png" alt="categories" width="1050" height = "400"/>

Some notable observations from the bar chart above:
- In 2021, the number of books related to **fiction & romance** and **family** has gone up. This could be due to the difficult times due to the pandemic which undoubtedly left lots of people with desires of affection and human contact. 
link: https://www.frontiersin.org/articles/10.3389/fpsyg.2021.798260/full
- There was a noticeable increase in quantity of books related to **student help** in 2012. This could be linked to the boom in popularity of massive open online courses - MOOCs in those years. It could be that the boom in MOOCs cause an increase in the production of texts for helping students in various aspects.  
link: https://onlinelearninginsights.wordpress.com/2012/12/21/what-the-heck-happened-in-2012-review-of-the-top-three-events-in-education/
- Not many books related to **sport & outdoors** are produced. This could be due to the fact that this is a theme in which people prefer to learn by doing rather than reading. 
- All the trends are decreasing and the sheer quantity of books are also less. This could be due to lack of data for these years or perhaps it could reflect a general trend of diminishing interest in books due to the internet. 


---

> # Sentiment Analysis

As mentioned before, we have a dataset of 3 million reviews, with an associated score from 1 to 5 stars. We will focus on trying to predict the sentiment of a review based on the review summary instead of the entire text itself. This is done mainly for computational reasons. 

We try to create wordclouds for positive and negative reviews, mainly to understand the type of sentiments we expect to find. One of the first difficulties is how to group 3 star reviews. In similar applications, it is customary to group 3 star reviews together with 4 and 5 stars and classify them as "positive". However, a quick sampling of 3 star reviews shows that they are associated to both negative and positive sentiments. Therefore, they were marked as "neutral", while 1-2 and 4-5 star reviews were classified as "negative" and "positive" respectively. 

We show the respective wordclouds:
<p float="left">
  <img src="./Images/negative_wordcloud.png" width="550" />
  <img src="./Images/positive_wordcloud.png" width="550" /> 
</p>

For the negative sentiment wordcloud (left) we have words such as "dissappointing", "bad", "boring", which indicate the emotions associated to these reviews. However, we also observe the presence of words such as "great" and "good", which seems counterintuitive at first glance. However, a quick search and find of the documents containing these terms shows that they are always preceded by the word "not" or "so", in order to form a negative sentiment. This could have been avoided through the use of joint collocations.

For the positive sentiment cloud, we see words such as "good", "great", "wonderful" and "excellent", which is a nice insight into the type of sentiments that are communicated with positive reviews. 

We proceed to the sentiment classification. As a first, most natural step, we use multilingual BERT for sentiment analysis since this model outputs a score from 1-5 and is trained on multiple languages. At first glance, this model seems to be perfect for our current needs and situation - scores from 1-5 and reviews in possibly different languages. 

---
> # Results and Discussion
