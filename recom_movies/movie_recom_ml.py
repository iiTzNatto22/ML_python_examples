import numpy as np 
from collections import defaultdict

data_path = r"C:/Users/Natto/Downloads/ml-latest-small/ratings.csv"
n_users = 6040
n_movies = 0
def load_rating_data(data_path, n_users, n_movies):
    """
    Load rating data from file and also return the number of ratings for each movie and movie_id index mapping
    @param data_path: path of the rating data file
    @param n_users: number of users
    @return: rating data in the numpy array of [user, movie]; movie_n_rating, {movie_id: number of ratings};
             movie_id_mapping, {movie_id: column index in rating data}
    """

    movie_id_mapping = {}
    movie_n_rating = defaultdict(int)
    with open(data_path, 'r') as file:
        max_movie_id = 0
        for line in file.readlines()[1:]:
            userID, movieID, rating, _ = line.split(',')
            userID = int(userID) - 1
            movieID = int(movieID)
            max_movie_id = max(max_movie_id, movieID)
            if movieID not in movie_id_mapping:
                movie_id_mapping[movieID] = len(movie_id_mapping)
            rating = float(rating)
            if rating > 0:
                movie_n_rating[movieID] += 1

        n_movies = max_movie_id + 1
        data = np.zeros([n_users, n_movies], dtype=np.float32)
        file.seek(0)
        for line in file.readlines()[1:]:
            userID, movieID, rating, _ = line.split(',')
            userID = int(userID) - 1
            movieID = int(movieID)
            rating = float(rating)
            data[userID, movie_id_mapping[movieID]] = rating 

    return(data, movie_n_rating, movie_id_mapping)

data, movie_n_rating, movie_id_mapping = load_rating_data(data_path, n_users, n_movies)


def display_distribution(data):
    values, counts = np.unique(data, return_counts=True)
    for value, count in zip(values, counts):
        print(f'Number of rating {int(value)}: {count}')

display_distribution(data)

movie_id_most, n_ratting_most = sorted(movie_n_rating.items(), key=lambda d: d[1], reverse=True)[0]
print(f'Movie ID {movie_id_most} has {n_ratting_most} ratings.')

X_raw = np.delete(data, movie_id_mapping[movie_id_most], axis=1)
Y_raw = data[:, movie_id_mapping[movie_id_most]]

X = X_raw[Y_raw > 0]
Y = Y_raw[Y_raw > 0]

print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)

display_distribution(Y)

recommend = 3 
Y[Y <= recommend] = 0
Y[Y > recommend] = 1 

n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()
print(f'{n_pos} positve samples and {n_neg} negative samples.')

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(len(Y_train), len(Y_test))

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)

# Prediction probabilities
prediction_prob = clf.predict_proba(X_test)
print(prediction_prob[0:10])

# Prediction class
prediction = clf.predict(X_test)
print(prediction[:10])

accuracy = clf.score(X_test, Y_test)
print(f'the accuracy is: {accuracy*100: 1f}%')

# Confusion Matrix 
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, prediction, labels=[0, 1]))

from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(Y_test, prediction, pos_label=1)
recall_score(Y_test, prediction, pos_label=1)
f1_score(Y_test, prediction, pos_label=1)

f1_score(Y_test, prediction, pos_label=0)

from sklearn.metrics import classification_report
report = classification_report(Y_test, prediction)
print(report)
