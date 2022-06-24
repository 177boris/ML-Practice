import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# try using lemmatization, test the difference in model accuracies 

class ReviewCleaner:
   
    @staticmethod
    def clean_review(review):
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review
                    if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        return review