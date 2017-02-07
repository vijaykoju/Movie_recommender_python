# #!/usr/bin/python2
""" Movie recommender
    Based on Andrew Ng's Machine Learning (Coursera) course
"""

import numpy as np
import pandas as pd
from scipy import optimize
import time
import matplotlib.pyplot as plt
from mvrec import *


# check the output of the cost and gradient functions.
def test():
    Y = pd.read_csv('movie_ratings_Y.txt', delimiter=',', header=None).values
    R = (Y != 0)*1
    print('Average rating for movie 1 (Toy Story): \
          {}'.format(np.mean(Y[0, np.where(R[0, :] == 1)[0]])))
    # plt.imshow(R)
    # plt.show()

    X = pd.read_csv('movie_param_X.txt', delimiter=',', header=None).values
    Theta = pd.read_csv('movie_param_Theta.txt', delimiter=',',
                        header=None).values

    num_users, num_movies, num_features = 4, 5, 3
    X = X[0:num_movies, 0:num_features]
    Theta = Theta[0:num_users, 0:num_features]
    Y = Y[0:num_movies, 0:num_users]
    R = R[0:num_movies, 0:num_users]

    init_param = np.concatenate((X.flatten('F'), Theta.flatten('F')))
    J = cofiCostFunc(init_param, Y, R, num_users, num_movies,
                     num_features, 1.5)
    grad = cofiGradFunc(init_param, Y, R, num_users, num_movies,
                        num_features, 1.5)
    print('Cost at loaded parameters (lambda = 1.5): {} \n \
        (this value should be about 31.34)'.format(J))
    print(grad)


# recommend new movies based on the prior rated movies.
def recommend_movies(eval_cost=True):
    movie_list = load_movies(filename='movie_ids.txt')

    # movie ratings for 1682 movies by 943 users.
    # rating scale 1-5. 0 indicates not rated.
    Y = pd.read_csv('movie_ratings_Y.txt', delimiter=',', header=None).values

    num_movies = Y.shape[0]  # number of movies

    # initiate my ratings
    my_ratings = np.zeros((num_movies, 1))
    my_ratings[0, 0] = 4
    my_ratings[6, 0] = 3
    my_ratings[11, 0] = 5
    my_ratings[53, 0] = 4
    my_ratings[63, 0] = 5
    my_ratings[65, 0] = 3
    my_ratings[68, 0] = 5
    my_ratings[97, 0] = 2
    my_ratings[182, 0] = 4
    my_ratings[225, 0] = 5
    my_ratings[354, 0] = 5

    print('New user ratings:')
    for i, rating in enumerate(my_ratings):
        if rating > 0:
            print('Rated {} for {}'.format(rating[0], movie_list[str(i+1)]))

    # add my_ratings to the overall rating matrix Y.
    Y = np.append(my_ratings, Y, axis=1)

    # binary matrix to identify which movies are rated and which are not
    # 1 --> rated, 0 --> not rated.
    R = (Y != 0) * 1

    # Normalize Ratings
    Y, Ymean = normalizeRatings(Y, R)

    # Useful values
    num_users = Y.shape[1]  # number of users
    num_features = 10       # number of movie features

    # movie parameters
    # X = pd.read_csv('movie_param_X.txt', delimiter=',', header=None).values
    # Theta = pd.read_csv('movie_param_Theta.txt', delimiter=',',
    #                     header=None).values
    # Theta = np.append(np.array([Theta[0, :]]), Theta, axis=0)
    X = np.random.randn(num_movies, num_features)
    Theta = np.random.randn(num_users, num_features)

    # convert X and Theta into vectors and combine them into one.
    # this is required because fmin_cg requires a vector.
    init_param = np.concatenate((X.flatten('F'), Theta.flatten('F')))

    lmbda = 10  # regularization parameter

    max_iter = 25
    # compute optimized parameters using fmin_cg.
    if eval_cost:
        theta = optimize.fmin_cg(cofiCostFunc, fprime=cofiGradFunc, x0=init_param,
                                 args=(Y, R, num_users, num_movies, num_features,
                                       lmbda), maxiter=max_iter, retall=True)
    else:
        theta = optimize.fmin_cg(cofiCostFunc, fprime=cofiGradFunc, x0=init_param,
                                 args=(Y, R, num_users, num_movies, num_features,
                                       lmbda), maxiter=max_iter, full_output=True)

    sol = theta[0]

    # unroll optimized values from fmin_cg back into X and Theta
    n = num_movies*num_features
    X = sol[0:n].reshape(num_movies, num_features, order='F')
    Theta = sol[n:].reshape(num_users, num_features, order='F')

    # new predictions (normalized)
    p = X.dot(Theta.T)

    # predictions of me based on my rated movies.
    my_predictions = p[:, 0] + Ymean[:, 0]

    # sort the predicted movies in descending order.
    sp, si = np.sort(my_predictions)[::-1], np.argsort(my_predictions)[::-1]

    # print 10 top most-rated movies for me.
    print('Recommendations for me: ')
    for i in range(10):
        print('Rated {:03.1f} for movie {}'.format(my_predictions[si[i]],
                                                   movie_list[str(si[i]+1)]))

    # evaluate and plot cost function
    if eval_cost:
        cost_fun = [cofiCostFunc(theta[1][i], Y, R, num_users, num_movies,
                                 num_features, lmbda) for i in range(max_iter)]
        plt.plot(range(max_iter), cost_fun)
        plt.show()


if __name__ == '__main__':
    start = time.time()
    # test()
    recommend_movies(eval_cost=False)
    end = time.time()
    print(end-start)
