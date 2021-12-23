import pandas as pd
import numpy as np
from user import user
from scipy import spatial, stats
import math



def clean_rating(rating):
    rating = int(np.rint(rating))
    if rating > 5:
        print(rating)
        rating = 5
    elif rating < 1:
        print(rating)
        rating = 1

    return rating


def clean_ratings(ratings):
    return [clean_rating(r) for r in ratings]
	
	
	
def train_users(users, train_file='train.txt'):
    training = open('train.txt', 'r')
    training = training.read().strip().split('\n')
    for i, line in enumerate(training):
        users[i] = [int(x) for x in line.split()]
		
		
		
	


def cosine_similarity(a, b):
    a_new, b_new = filter_common(a, b)

    sim = np.dot(a_new, b_new)

    norm_a = vector_norm(a_new)
    norm_b = vector_norm(b_new)
    if norm_a != 0 and norm_b != 0:
        sim /= (norm_a * norm_b)
    else:
        sim = 0

    if sim > 1:
        sim = 1
    elif sim < -1:
        sim = -1

    return sim
	
def test_dataset(users, dataset_file):
    dataset = open(dataset_file, 'r').read().strip().split('\n')
    dataset = [data.split() for data in dataset]
    dataset = [[int(e) for e in data] for data in dataset]
    current_user_id = dataset[0][0] - 1
    current_user = {}
    movie_ids = []
    results = []
    for user_id, movie_id, rating in dataset:
        user_id -= 1
        movie_id -= 1
        if user_id != current_user_id:
            process_stored_data(
                users,
                current_user,
                current_user_id,
                movie_ids,
                results
            )
            current_user_id = user_id
            current_user = {}
            movie_ids = []

        if rating == 0:
            movie_ids.append(movie_id)
        else:
            current_user[movie_id] = rating

    process_stored_data(
        users,
        current_user,
        current_user_id,
        movie_ids,
        results
    )

    return results


	

def score_batch_item_centered(users, user, user_id, movie_ids):
    items = np.array(users).T
    ratings = []
    user_items = list(user.keys())
    user_averages = [np.average([r for r in u if r > 0]) for u in users]

    for movie_id in movie_ids:
        item = items[movie_id]
        i_ratings = [r for r in item if r > 0]
        if len(i_ratings) > 0:
            r_avg = np.average(i_ratings)
        else:
            r_avg = 3

        weights = [adj_cosine_similarity(items[i], item, users)
                   for i in user_items]
        sum_w = 0
        rating = 0

        for w, i, user_avg in zip(weights, user_items, user_averages):
            u_rating = user[i]
            sum_w += np.abs(w)
            rating += (w * (u_rating - user_avg))

        if sum_w != 0:
            rating = r_avg + (rating/sum_w)
        else:
            rating = r_avg

        rating = int(np.rint(rating))
        ratings.append(rating)

    return clean_ratings(ratings)

def process_stored_data(users, user, user_id, movie_ids, results):
    if len(movie_ids) > 0:
        ratings = score_batch_item_centered(users, user, user_id, movie_ids)

        for m_id, r in zip(movie_ids, ratings):
            if r < 1 or r > 5:
                raise Exception('Rating %d' % r)
            results.append((user_id+1, m_id+1, r))

def adj_cosine_similarity(a, b, users):
    if not hasattr(adj_cosine_similarity, 'avgs'):
        filtered_users = [
            [x for x in u if x > 0] for u in users]
        adj_cosine_similarity.avgs = [np.mean(u) for u in filtered_users]

    avgs = adj_cosine_similarity.avgs
    a_adj = np.subtract(a, avgs)
    b_adj = np.subtract(b, avgs)
    a_new, b_new = filter_common(a_adj, b_adj)

    return cosine_similarity(a_new, b_new)
	
def filter_common(v1, v2):
    v1_new = []
    v2_new = []
    for i, x in enumerate(v1):
        y = v2[i]
        if y > 0 and x > 0:
            v1_new.append(x)
            v2_new.append(y)

    return np.array(v1_new), np.array(v2_new)


def vector_norm(v):
    return np.sqrt(np.sum(np.square(v)))


def log_results(results, logfile):
    fout = open(logfile, 'w')
    for result in results:
        fout.write(' '.join(str(x) for x in result) + '\n')


def test_all(users):

    results = test_dataset(users, 'test5.txt')
    log_results(results, 'cosine_itemtest5.txt')

    results = test_dataset(users, 'test10.txt')
    log_results(results, 'cosine_itemtest10.txt')

    results = test_dataset(users, 'test20.txt')
    log_results(results, 'cosine_itemtest20.txt')


def main():
    num_users = 200
    num_movies = 1000
    users = [[0] * num_movies] * num_users
    train_users(users, 'train.txt')
    test_all(users)

main()





