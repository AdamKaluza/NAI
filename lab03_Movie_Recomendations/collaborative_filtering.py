"""
==========================================
Program to recommend movies which user should/ should not watch
Creators:
Adam Kałuża s20831
Krzysztof Lewandowski s20491
==========================================
To run program install:
pip install numpy
==========================================
Usage:
You need to run collaborative_filtering.py with --user "USER" flag
python collaborative_filtering.py --user "Paweł Czapiewski"
In output you will get:
- five recommended movies
- five not recommended movies
==========================================
"""

import argparse
import json
import numpy as np

from compute_scores import euclidean_score


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find users who are similar to the input user')
    parser.add_argument('--user', dest='user', required=True,
                        help='Input user')
    return parser

"""
Finds users in the dataset that are similar to the input user
"""
def find_similar_users(dataset, user, num_users):
    if user not in dataset:
        raise TypeError('Cannot find ' + user + ' in the dataset')

    """
    Compute Pearson score between input user
    and all the users in the dataset
    """
    scores = np.array([[x, euclidean_score(dataset, user,
                                           x)] for x in dataset if x != user])

    """
    Sort the scores in decreasing order
    """
    scores_sorted = np.argsort(scores[:, 1])[::-1]

    """
    Extract the top 'num_users' scores
    """
    top_users = scores_sorted[:num_users]

    return scores[top_users]


def add_similar_users_to_list(database, similars):
    """
    saves similar user's names to list
    """
    for item in database:
        for sim_user in similars:
            if item == sim_user[0]:
                similars_users_names.append(item)


def create_films_set():
    """
    add films from each similar user
    """
    for name in similars_users_names:
        save_films(data[name])

    """
    we need to remove lecturer seen movies
    """
    for paul_film in data[user]:
        films.remove(paul_film)


def find_recommended_films():
    """
    initialize variables for describing recommended films
    """
    score_index = 10

    """
    going from the top rating, we are looping through all the films
    we check if users watched shared films and if so, we check their ratings,
    if it's current searched top rating, we add it to recommended films
    """
    while score_index > 0:
        for film in films:
            for name in similars_users_names:
                if len(recommended_films) < 5:
                    if film in data[name] and data[name][film] == score_index:
                        recommended_films.add(film)
        score_index -= 1


def find_not_recommended_films():
    """
    initialize variables for describing recommended films
    """
    score_index = 1

    """
    going from the top rating, we are looping through all the films
    we check if users watched shared films and if so, we check their ratings,
    if it's current searched top rating, we add it to recommended films
    """
    while score_index < 11:
        for film in films:
            for name in similars_users_names:
                if len(not_recommended_films) < 5:
                    if film in data[name] and data[name][film] == score_index:
                        not_recommended_films.add(film)
        score_index += 1

"""
function to save films to set
"""
def save_films(user_data):
    for film in user_data:
        films.add(film)


"""
function to pint films
"""
def print_movies(movies):
    rank = 1
    for movie in movies:
        print("{}: {}".format(rank, movie))
        rank = rank + 1


"""
function to calculate users distance
"""
def calculate_user_distance():
    i = len(similar_users) - 1
    while i > 0:
        x = float(similar_users[0][1])
        y = float(similar_users[i][1])
        if x - acceptable_diff > y:
            user_dist.append(similar_users[0])
        else:
            user_dist.append(similar_users[i])
        i -= 1


if __name__ == '__main__':
    """
    we dont want repeated films
    """
    recommended_films = set()
    not_recommended_films = set()
    acceptable_diff = 0.1

    similars_users_names = []
    user_dist = []
    films = set()
    args = build_arg_parser().parse_args()
    user = args.user

    ratings_file = 'ratings.json'

    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())

    save_films(data[user])
    similar_users = find_similar_users(data, user, 3)
    calculate_user_distance()
    add_similar_users_to_list(data, similar_users)
    create_films_set()
    find_recommended_films()
    find_not_recommended_films()

    print('Recommended films: ')
    print_movies(recommended_films)
    print('\nNot recommended films: ')
    print_movies(not_recommended_films)
