from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s
from unidecode import unidecode

# consumer key, consumer secret, access token, access secret.
ckey = "n9zcsi8C5jjo9u5tlSSRk9vVn"
csecret = "RK6iQsR9ungazOXLUEIcIvNBBfUW6NMNzGRAKjMaQPxdZgFEeg"
atoken = "1262553171349475328-02S35lIrZ5nCmoVdRX9HKZS3aqOXUs"
asecret = "ASRzAX5vuuQhJEmCeIbDAa2sGSoFTFGFBJ7PEgUkztheb"


# from twitterapistuff import *

class listener(StreamListener):

    def on_data(self, data):

        all_data = json.loads(data)
        try:
            tweet = unidecode(all_data['text'])
            sentiment_value, confidence = s.sentiment(tweet)
            if sentiment_value == 0:
                sentiment_value = 'Tweet negativo'
            else:
                sentiment_value = 'Tweet positivo'
            print(tweet, sentiment_value, confidence * 100)

            if confidence * 100 >= 80:
                output = open("twitter-out.txt", "a")
                output.write(sentiment_value)
                output.write('\n')
                output.close()
        except:
            pass

        return True

    def on_error(self, status):
        print(status)


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["Trump"])
