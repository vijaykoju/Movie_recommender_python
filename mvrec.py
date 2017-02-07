import numpy as np
import codecs


# Load movies into a dict from 'filename'
def load_movies(filename):
    """ Load movies into a dict from 'filename' """
    movie_list = {}
    with codecs.open(filename, 'r', encoding="ISO-8859-1") as f:
        for line in f:
            line = line.split()
            movie_list[line[0]] = " ".join(line[1:])

    return movie_list


# Normalize movie ratings.
def normalizeRatings(Y, R):
    """ Normalize movie ratings. """
    m, n = Y.shape
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros((m, n))
    for i in range(m):
        idx = np.where(R[i, :] == 1)[0]
        Ymean[i, 0] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i, 0]

    return Ynorm, Ymean


# Compute cost function for optimization.
def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lmbda):
    """ Compute cost function for optimization. """
    n = num_movies*num_features
    X = params[0:n].reshape(num_movies, num_features, order='F')
    Theta = params[n:].reshape(num_users, num_features, order='F')

    J = 0.5*np.sum(((X.dot(Theta.T) - Y)**2)*R) + \
        0.5*lmbda*(np.sum(Theta**2) + np.sum(X**2))

    return J


# Compute gradient for optimization.
def cofiGradFunc(params, Y, R, num_users, num_movies, num_features, lmbda):
    """ Compute gradient for optimization. """
    n = num_movies*num_features
    X = params[0:n].reshape(num_movies, num_features, order='F')
    Theta = params[n:].reshape(num_users, num_features, order='F')

    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    tmp = (X.dot(Theta.T) - Y)*R
    X_grad = tmp.dot(Theta) + lmbda*X
    Theta_grad = tmp.T.dot(X) + lmbda*Theta

    grad = np.concatenate((X_grad.flatten('F'), Theta_grad.flatten('F')))

    return grad
