import pandas as pd
import numpy as np
import pickle
import json


dataset = pd.read_csv('cleared_reviews.csv')


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(dataset["review"].values.astype('U')).toarray()
y = dataset["is_positive"]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 400,
                            criterion = 'entropy')


model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Score: ', model.score(X_test, y_test))


# Persists the model 
words = cv.vocabulary_
words_for_json = {}
for k, v in words.items():
    words_for_json[k] = int(v)
pickle.dump(model, open('reviewClassifier.pkl','wb'))
with open('word_feature_space.json', 'w') as fp:
    json.dump(words_for_json, fp)
    