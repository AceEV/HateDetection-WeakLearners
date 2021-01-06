import tweepy
import pandas as pd
import json

# Set your ceredntials
API_KEY = ''
API_SECRET = ''
BEARER_TOKEN = ''
ACCESS_TOKEN = ''
ACESS_TOKEN_SECRET = ''


def fetch_from_twitter(ids, api, savefile="dump.csv"):
    # Set the columns that you want to retrieve
    data = pd.DataFrame([], columns=["id", "text", "user", "user_verified", "user_followers_count",
                                     "user_friends_count", "retweet_count", "fav_count", "hashtags"])

    # The API supports a max of 100 tweets in a single call.
    # Itereate through ids[] 100 at a time
    for i in range(0, len(ids[:200]), 100):
        x = [i for i in range(i, i+100)]
        print("Currently fething for", x[0], "-->", x[-1])
        list_of_100_tweets = ids[i:i+100]
        # print(list_of_100_tweets)
        tweets = api.statuses_lookup(list_of_100_tweets)
        for t in tweets:
            tj = t._json
            tweet_data = {
                "id": tj["id"],
                "text": tj["text"],
                "user": tj["user"]["screen_name"],
                "user_verified": tj["user"]["verified"],
                "user_followers_count": tj["user"]["followers_count"],
                "user_friends_count": tj["user"]["friends_count"],
                "retweet_count": tj["retweet_count"],
                "fav_count": tj["favorite_count"],
                "hashtags": tj["entities"]["hashtags"]
            }

            data = data.append(tweet_data, ignore_index=True)

    # Save to a csv
    data.to_csv(savefile)


if __name__ == '__main__':

    ids = []
    # Use any method to bring a list of tweets into the ids[]

    auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True,
                     wait_on_rate_limit_notify=True)

    with open("data/random_tweet_ids.json") as json_file:
        ids = json.load(json_file)

    fetch_from_twitter(ids, api, "unlabeled_data_dump.csv")


# NOTE : If the tweet is deleted by Twitter themselves, those rows are not present in the df
