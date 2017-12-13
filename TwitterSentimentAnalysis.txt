import tweepy
import requests
import urllib.request
import urllib.parse
import re
import csv 
from textblob import TextBlob 

consumer_key = 'CPk3GUu7FWTn5RddBJaYFFy8K'
consumer_secret = '0TBRdEV1ygdD3KCdMtx40F4eKsVIZcpEHlYyXNsW18btKhZvq6'

access_token = '937989586457911296-Xcz1GHvFgG1Fltx887C2fY81kKnZMsN'
access_token_secret = 'JDJ14z5cYzBZJ6LBxZAcHzAyyvjrSnK41IU2lv6SAC9OX'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Apple INC',  since="2017-12-03",
    until="2017-12-04",
 lang = 'en')

sumTwitter = 0
tweets = 0
for tweet in public_tweets: 
    tweets+= 1
    print(tweet.created_at)
    #print(tweet.text)
    analysis = TextBlob(tweet.text)
    sumTwitter += analysis.sentiment.polarity

averageSum = (sumTwitter/tweets)    

sumWeb = 0

url = 'https://techcrunch.com/2017/12/11/apple-knockoff-myetherwallet-ios/'
values = {'s':'basics',
          'submit':'search'}
data = urllib.parse.urlencode(values)
data = data.encode('utf-8')
req = urllib.request.Request(url, data)
resp = urllib.request.urlopen(req)
respData = resp.read()

paragraphs = re.findall(r'<p>(.*?)</p>',str(respData))

for eachP in paragraphs:
    print(eachP)
    analysis = TextBlob(eachP)
    sumWeb += analysis.sentiment.polarity

totalSum = (sumWeb+averageSum)/2

print(totalSum)