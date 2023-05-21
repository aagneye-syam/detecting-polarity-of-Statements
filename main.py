from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer

positive_set = [
    "we are lucky",
    "we are happy",
    "we are loved",
    "we are good",
]

negative_set = [
    "we are unlulcky",
    "we are hated",
    "we are unhappy",
    "we are bad",
]

sample_set = [
    "we are lucky",
    "we are unluky",
    "we are happy",
    "i am bad",
    "we are unhappy",
]

print(sample_set)

data_set = positive_set +negative_set
data_labels = ["POSITIVE"]*len(positive_set) + ["NEGATIVE"]*len(negative_set)

vectorizer = CountVectorizer()
vectorizer.fit(data_set)
data_vector = vectorizer.transform(data_set)
sample_vector = vectorizer.transform(sample_set)
feature_name = vectorizer.get_feature_names_out()
#print(data_vector.toarray())
#print("        ")
#print(sample_vector.toarray())
#print(feature_name)

classifier = tree.DecisionTreeClassifier()
classifier.fit(data_vector, data_labels)
predictions = classifier.predict(sample_vector)
print(predictions)
