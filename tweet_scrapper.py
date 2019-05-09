
import tweepy
#from textblob import TextBlob
#from sqlalchemy.exc import ProgrammingError
from tweepy import OAuthHandler
from pymongo import MongoClient
import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



import json



analyzer = SentimentIntensityAnalyzer()
def senti_analyzer_label(text):
    score = analyzer.polarity_scores(text)
    #score = 
    lb = score['compound']
    if lb >= 0.05:
        return 'Positive'
    elif (lb > -0.05) and (lb < 0.05):
        return 'Neutral'
    else:
        return 'Negative'


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)        
    return input_txt


def clean_tweets(text):
    # remove twitter Return handles (RT @xxx:)
    text = np.vectorize(remove_pattern)(text, "RT @[\w]*:")
    # remove twitter handles (@xxx)
    text = np.vectorize(remove_pattern)(text, "@[\w]*")
    # remove URL links (httpxxx)
    text = np.vectorize(remove_pattern)(text, "https?://[A-Za-z0-9./]*")
    # remove special characters, numbers, punctuations (except for #)
    text = np.core.defchararray.replace(text, "[^a-zA-Z#]", " ")
    return text

client = MongoClient('localhost', 27017)
db = client['admin']

class StreamListener(tweepy.StreamListener):

    def on_status(self, status):
        if status.retweeted:
            return
        if hasattr(status, 'retweeted_status'):
            try:
                text = status.retweeted_status.extended_tweet["full_text"]
            except:
                text = status.retweeted_status.text
        else:
            try:
                text = status.extended_tweet["full_text"]
            except AttributeError:
                text = status.text     
        description = str(status.user.description)
        loc = status.user.location
        text = text.lower()
        text = str(clean_tweets(text))
        name = status.user.screen_name
        user_created =status.user.created_at
        followers = status.user.followers_count
        id_str = status.id_str
        created = status.created_at
        retweets = status.retweet_count
        #blob = TextBlob(text)
        #sent = blob.sentiment

    
       
        print(text,senti_analyzer_label(text))      
                
        
       # table = db[TABLE_NAME]
        try:
            collection = db['elect']
            data = {
                'user_description'    :         description,
                'user_location'   :     loc,
                'text' :    text,
                'user_name' :    name,
                'user_created' :    '{:%m/%d/%Y}'.format(user_created),
                'user_followers' :    followers,
                'id_str' :    id_str,
                'created' :    '{:%m/%d/%Y}'.format(created),
                'retweet_count' :    retweets,
                'polarity' :    analyzer.polarity_scores(text),
                'label' :    senti_analyzer_label(text)}
            
            
            
            collection.insert(data)
        except ProgrammingError as err:
            print(err)

    def on_error(self, status_code):
        if status_code == 420:
            #returning False in on_data disconnects the stream
            return False
        
        




        

consumer_key, consumer_secret = 'XE5F8IGwN4LSwlM2oKdlRftk7', 'F2xBRiWlliyc1yLTHD8s3E5gJeXZBFYYC1H9oWIFaa1Tk716mG'
access_token, access_token_secret = '1004136301573099520-NG78DBmsrpnyDU2C8lRk7j9hMnbnt3','151xLtzobJo8O3uma9Z80uylyXPJB0ARaLzMZLjm5rGIW'
auth = OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth)

stream_listener = StreamListener()

def start_str():
    while True:
        try:
            stream = tweepy.Stream(auth=api.auth, listener=stream_listener, tweet_mode = 'extended')
            keys = ['pdp', 'apc', 'buhari','atiku', 'pdpAtiku', 'apcbuhariu', 'nigeriadecides', 'nigeriadecides2019', 'muhamadu buhari', 'atiku abubakar', 'pdpnigeria', 'apcnigeria', 'atiku peterobi', 'peterobi', 'donaldduke','sowore', 'rescuenigeria', 'fight4naija']
            stream.filter(track= keys)

        except:
            continue


start_str()
