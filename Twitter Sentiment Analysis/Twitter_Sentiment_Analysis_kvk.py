import os
os.getcwd()
os.chdir("D:/Kaushik/Uconn Related/Study and Research/My Projects/Twitter")

import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
import pandas as pd
import datetime 
import MyCredentials_kvk as MyCred


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

import pyodbc

class TwitterClient(object):
    '''
    Generic Twitter Class for sentiment analysis.
    '''
    def __init__(self):
        '''
        Class constructor or initialization method.
        '''
        
        consumer_key= twitter_consumer_key
        consumer_secret=  twitter_consumer_secret
        access_token=  twitter_access_token
        access_token_secret= twitter_access_token_secret

        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")
 
    def clean_tweet(self, tweet):
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
 
    def get_tweet_sentiment(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.clean_tweet(tweet))
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'
    
            
    def get_tweet_subjectivity(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.clean_tweet(tweet))
        return round(analysis.sentiment.subjectivity,2)
        """
        if analysis.sentiment.subjectivity == 1:
            return 'positive'
        elif analysis.sentiment.subjectivity == 0:
            return 'neutral'
        else:
            return 'negative'
        """
        
    def get_tweet_corrected(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.clean_tweet(tweet))
        return analysis.correct()
        """
        if analysis.sentiment.subjectivity == 1:
            return 'positive'
        elif analysis.sentiment.subjectivity == 0:
            return 'neutral'
        else:
            return 'negative'
        """
        
        
    def get_tweets(self, query, count = 10):
        '''
        Main function to fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        tweets = []
        curr_time = datetime.datetime.now()
        datetime_format = curr_time.strftime("%Y_%m_%d_%H_%M_%S")
        file_name = "export_tweets_" + datetime_format + ".csv"
             
    
        try:
            # call twitter api to fetch tweets
            fetched_tweets = self.api.search(q = query, count = count)
                     
            tweets_data = pd.DataFrame(columns=('created_at', 'text', 'retweet_count', 'sentiment'))  
            
            # parsing tweets one by one
            for tweet in fetched_tweets:
                # empty dictionary to store required params of a tweet
                if(tweet.lang == "en"):
                    parsed_tweet = {}
                    parsed_tweet['created_at'] = tweet.created_at
                    parsed_tweet['lang'] = tweet.lang
                    parsed_tweet['text'] = tweet.text.encode("utf-8")
                    parsed_tweet['retweet_count'] = tweet.retweet_count
                    parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
                    parsed_tweet['subjectivity'] = self.get_tweet_subjectivity(tweet.text)
                    #parsed_tweet['corrected'] = self.get_tweet_corrected(tweet.text)
                    
                    
                    # appending parsed tweet to tweets list
                    if tweet.retweet_count > 0:
                        # if tweet has retweets, ensure that it is appended only once
                        if parsed_tweet not in tweets:
                            tweets.append(parsed_tweet)
                            tweets_data  = tweets_data.append(parsed_tweet, ignore_index=True)
                    else:
                        tweets.append(parsed_tweet)
                        tweets_data  = tweets_data.append(parsed_tweet, ignore_index=True)
                       
            tweets_data.to_csv(file_name, index=False)
            # return parsed tweets
            return tweets, file_name
 
        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))
        
            
    def send_email( body, file_name):                        
        msg = MIMEMultipart()
        
        subject = "Python - Twitter Sentiment Analysis"
        msg['From'] = gmail_id
        msg['To'] = gmail_id
        msg['Subject'] = subject
            
        msg.attach(MIMEText(body,'plain'))
            
        attachment = open(file_name,'rb')
            
        part = MIMEBase('application','octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',"attachment; filename= "+ file_name)
            
        msg.attach(part)
        text = msg.as_string()
        server = smtplib.SMTP('smtp.gmail.com',587)
        server.starttls()
        server.login(gmail_id,gmail_password)
            
            
        server.sendmail(gmail_id,gmail_id,text)
        attachment.close()
        server.quit()

        
        
    def export_tweets_sql():
        
        cnxn = pyodbc.connect(r'Driver={SQL Server};Server=.\SQLEXPRESS;Database=personal_kvk;Trusted_Connection=yes;')
        cursor = cnxn.cursor()
        cursor.execute("SELECT * FROM dbo.Department")
        while 1:
            row = cursor.fetchone()
            if not row:
                break
            print(row.DepartmentName)
        cnxn.close()




def main():
    twitter_search_criteria = "Obama"
      
            
    # creating object of TwitterClient Class
    api = TwitterClient()
    # calling function to get tweets
    tweets,file_name = api.get_tweets(query = twitter_search_criteria, count = 100)
    
    # picking positive tweets from tweets
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
    # percentage of positive tweets
    positive_tweets_perc = "\nPositive tweets percentage: {} %".format(100*len(ptweets)/len(tweets))
    # picking negative tweets from tweets
    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    # percentage of negative tweets
    negetive_tweets_perc = "\nNegative tweets percentage: {} %".format(100*len(ntweets)/len(tweets))
    
    # percentage of neutral tweets
    #print("Neutral tweets percentage: {} %".format(100*len(tweets - ntweets - ptweets)/len(tweets)))
    greeting = "Hello Guys! \n\nThe Twitter Sentiment Analysis for the keyword " + "'" + twitter_search_criteria + "'" + " is completed!! \n"
    # printing first 5 positive tweets
    i = 1
    positive_tweet_body = "\n\nBelow are the top positive tweets: \n"
    for tweet in ptweets[:10]:
        positive_tweet_body += "\n" + i.__str__() + ". " + tweet['text']        
        i+=1
        
    # printing first 5 negative tweets
    i = 1
    negative_tweet_body =  "\n\nBelow are the top negative tweets: \n"
    for tweet in ntweets[:10]:
        negative_tweet_body +=  "\n" + i.__str__() + ". " + tweet['text']
        i+=1
        
    body = greeting + positive_tweets_perc +  negetive_tweets_perc + positive_tweet_body + negative_tweet_body
    send_email(body, file_name)
    
if __name__ == "__main__":
    # calling main function
    main()