import pandas as pd
import numpy as np
import re 
from collections import Counter
from review_cleaner import ReviewCleaner as rc

reviews_df = pd.read_csv("reviews.csv")

for index, row in reviews_df.iterrows():
    cleared_review = rc.clean_review(row['review'])
    reviews_df.at[index, 'review'] = cleared_review

most_common_100_words = Counter(" ".join(reviews_df["review"]).split()).most_common(100)

print(f"Most common words:  {most_common_100_words}")

unwanted_words = ['room', 'staff', 'locat', 'hotel', 'breakfast', 'bed', 'shower']  # from most common words...

reviews_df['review'] = reviews_df['review'].str.replace('|'.join(map(re.escape, unwanted_words)), '')

# Save the cleaned data as cleared_reviews.csv
endResult = reviews_df.to_csv('cleared_reviews.csv')

 
print("Reviews")
print(reviews_df.head())
