#!/usr/bin/env python
# coding: utf-8

# # **Sentiment-based product recommendation system**
# ### Capstone Project By : Janarthanan Mani

# ## To include below tasks:
# 
# 1. Data sourcing and sentiment analysis
# 2. Building a recommendation system
# 3. Improving the recommendations using the sentiment analysis model
# 4. Deploying the end-to-end project with a user interface

# ## Problem Statement 
# 
# The e-commerce business is quite popular today. Here, you do not need to take orders by going to each customer. A company launches its website to sell the items to the end consumer, and customers can order the products that they require from the same website. Famous examples of such e-commerce companies are Amazon, Flipkart, Myntra, Paytm and Snapdeal.
# 
#  
# 
# Suppose you are working as a Machine Learning Engineer in an e-commerce company named 'Ebuss'. Ebuss has captured a huge market share in many fields, and it sells the products in various categories such as household essentials, books, personal care products, medicines, cosmetic items, beauty products, electrical appliances, kitchen and dining products and health care products.
# 
#  
# 
# With the advancement in technology, it is imperative for Ebuss to grow quickly in the e-commerce market to become a major leader in the market because it has to compete with the likes of Amazon, Flipkart, etc., which are already market leaders.
# 
#  
# 
# As a senior ML Engineer, you are asked to build a model that will improve the recommendations given to the users given their past reviews and ratings. 

# ## Importing the necessary libraries

# In[ ]:

def product_predict(user):
    import numpy as np
    import pandas as pd
    import nltk
    import joblib
    import warnings
    warnings.filterwarnings("ignore")

    nltk.download('wordnet','./nltk_data/')
    nltk.download('stopwords','./nltk_data/')
    nltk.download('punkt','./nltk_data/')


    # # Loading Data
    sentiment = pd.read_csv('./data/sentiment_premodel.csv')
    sentiment.user_sentiment=sentiment.user_sentiment.astype('int')
    sentiment.unique_id=sentiment.unique_id.astype('int')
    sentiment.reviews_rating=sentiment.reviews_rating.astype('int')

    sentiment['user_sentiment'].value_counts()

    #Write your code here to initialise the TfidfVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    # Transform word vector in tfidf vector
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2,ngram_range=(2,2))
    X_train_tfidf = vectorizer.fit_transform(sentiment['reviews'])

    #Creating unique id's for user and product
    user_dict={}
    j=1
    for i in sentiment['reviews_username'].unique():
      user_dict[i]=j
      j=j+1

    prd_dict={}
    j=1
    for i in sentiment['name'].unique():
      prd_dict[i]=j
      j=j+1

    #Creating new columns to the dataframe with unique id's
    sentiment['product_id']=sentiment['name'].map(prd_dict)
    sentiment['user_id']=sentiment['reviews_username'].map(user_dict)

    sentiment['reviews_rating']=sentiment['reviews_rating'].astype('float')

    #Creating distinct user and product dataframe
    productdf=sentiment.groupby(['product_id','name']).size().reset_index(name='NoOfReviews')
    userdf=sentiment.groupby(['user_id','reviews_username']).size().reset_index(name='NoOfReviews')

    usercheck=userdf[userdf.reviews_username.isin(user)].empty

    if usercheck == True:
        user_invalid= pd.DataFrame({"Invalid_Username":['User does not Exist, Please search with valid user']})
        user_invalid=user_invalid.set_index('Invalid_Username')
        return user_invalid
    else:
        user_input=userdf[userdf.reviews_username.isin(user)].user_id.iloc[0]

    # Pivot the sentiment ratings' dataset into matrix format in which columns are products and the rows are user IDs.
    df_pivot = sentiment.pivot_table(
        index='user_id',
        columns='product_id',
        values='reviews_rating'
    ).fillna(0)

    # ### Creating dummy train
    # These dataset will be used for prediction
    # - Dummy train will be used later for prediction of the products which has not been rated by the user. To ignore the products rated by the user, we will mark it as 0 during prediction. The products not rated by user is marked as 1 for prediction in dummy train dataset.
    # Copy the train dataset into dummy_train
    dummy_train = sentiment.copy()

    # The movies not rated by user is marked as 1 for prediction.
    dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)

    # Convert the dummy train dataset into matrix format.
    dummy_train = dummy_train.pivot_table(
        index='user_id',
        columns='product_id',
        values='reviews_rating'
    ).fillna(1)

    # **Cosine Similarity**
    #
    # Cosine Similarity is a measurement that quantifies the similarity between two vectors [Which is Rating Vector in this case]
    #
    # **Adjusted Cosine**
    #
    # Adjusted cosine similarity is a modified version of vector-based similarity where we incorporate the fact that different users have different ratings schemes. In other words, some users might rate items highly in general, and others might give items lower ratings as a preference. To handle this nature from rating given by user , we subtract average ratings for each user from each user's rating for different products.
    #
    #

    # # User Similarity Matrix

    # ## Using Cosine Similarity

    from sklearn.metrics.pairwise import pairwise_distances

    # Creating the User Similarity Matrix using pairwise_distance function.
    user_correlation = 1 - pairwise_distances(df_pivot, metric='cosine')
    user_correlation[np.isnan(user_correlation)] = 0


    # ## Using adjusted Cosine

    # ### Here, we are not removing the NaN values and calculating the mean only for the product rated by the user

    # Create a user-product matrix.
    df_pivot = sentiment.pivot_table(
        index='user_id',
        columns='product_id',
        values='reviews_rating'
    )


    # ### Normalising the rating of the product for each user around 0 mean

    mean = np.nanmean(df_pivot, axis=1)
    df_subtracted = (df_pivot.T-mean).T

    # ### Finding cosine similarity

    from sklearn.metrics.pairwise import pairwise_distances

    # Creating the User Similarity Matrix using pairwise_distance function.
    user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
    user_correlation[np.isnan(user_correlation)] = 0

    # ## Prediction - User User

    # Doing the prediction for the users which are positively related with other users, and not the users which are negatively related as we are interested in the users which are more similar to the current users. So, ignoring the correlation for values less than 0.

    user_correlation[user_correlation<0]=0

    # Rating predicted by the user (for products rated as well as not rated) is the weighted sum of correlation with the product rating (as present in the rating dataset).

    user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))

    # Since we are interested only in the product not rated by the user, we will ignore the product rated by the user by making it zero.

    user_final_rating = np.multiply(user_predicted_ratings,dummy_train)

    d = user_final_rating.loc[user_input].T.reset_index().sort_values(by=user_input,ascending=False)[0:20]

    user_user = pd.merge(d[['product_id']],productdf,on='product_id', how = 'inner')

    # # Recommending top 5 products based on best performing reccommendation system
    #
    # ### In our case it is User-User Recommendation system with low RMSE 2.42

    prod_prediction=sentiment[sentiment.product_id.isin(user_user.product_id)]

    # ### Predicting sentiment for top 20 products based on best performing model Naive Bayes

    #Testing model from file
    bnb_file = joblib.load("./models/bnb_model.sav")

    final_pred=vectorizer.transform(prod_prediction['reviews'])

    prod_prediction['sentiment_predicted'] = bnb_file.predict(final_pred)

    # ### Predicting top 5 products to the user based on sentiment model

    final_prediction=prod_prediction.groupby(by=['name','sentiment_predicted'])[['name']].count()/prod_prediction.groupby(by='name')[['name']].count()*100

    final_prediction=final_prediction.rename(columns={'name':'percent'})
    final_prediction.reset_index(inplace=True)

    top_5_products=final_prediction[final_prediction.sentiment_predicted==1].sort_values('percent',ascending=False)[0:5][['name']]

    predicted_products=top_5_products[['name']]

    predicted_products=predicted_products.rename(columns={'name': 'top5products for user: '+str(user[0])}).set_index('top5products for user: '+str(user[0]))

    return predicted_products