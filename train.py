# import pandas as pd
# import re
# import pickle
# import emoji
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression

# df = pd.read_csv("Twitter_Data.csv")
# df = df[['clean_text', 'category']]
# df.dropna(inplace=True)

# def clean(t):
#     t = t.lower()
#     t = emoji.demojize(t)
#     t = re.sub(r'http\S+', '', t)
#     t = re.sub(r'@\w+', '', t)
#     t = re.sub(r'#(\w+)', r'\1', t)
#     t = t.replace(":", " ")
#     t = re.sub(r'[^a-z\s]', '', t)
#     return t

# df['clean_text'] = df['clean_text'].apply(clean)
# X = df['clean_text']
# y = df['category']
# tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
# X_vec = tfidf.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# model = LogisticRegression(max_iter=200)
# model.fit(X_train, y_train)

# print("Accuracy:", model.score(X_test, y_test))
# pickle.dump(model, open("model.pkl", "wb"))
# pickle.dump(tfidf, open("tfidf.pkl", "wb"))
import pandas as pd
import re
import pickle
import emoji
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Twitter_Data.csv")

df = df[['clean_text', 'category']]
df.dropna(inplace=True)

def clean(t):
    t = t.lower()
    t = emoji.demojize(t)
    t = re.sub(r'http\S+', '', t)
    t = re.sub(r'@\w+', '', t)
    t = re.sub(r'#(\w+)', r'\1', t)
    t = t.replace(":", " ")
    t = re.sub(r'(.)\1+', r'\1\1', t)
    t = re.sub(r'[^a-z\s]', '', t)
    return t

df['clean_text'] = df['clean_text'].apply(clean)

df = df[df['clean_text'].str.len() > 10]

X = df['clean_text']
y = df['category']

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    #stop_words='english'
)

X_vec = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = LogisticRegression(
    max_iter=300,
    class_weight='balanced'
)

model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))