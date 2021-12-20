import numpy as np
import pandas as pd
from typing import Optional
import plotly.express as px
import plotly.graph_objects as go


def neg_squared_euc_dists(ary: np.ndarray):
    """
    Compute matrix containing negative squared euclidean distance for all pairs of points in input matrix X
    (Implementation based around the following: https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/)
    (Math? See https://stackoverflow.com/questions/37009647)

    :param ary: A matrix of size NxD
    :return: An NxN matrix D, with entry D_ij = negative squared euclidean distance between rows X_i and
    """
    #
    ary_sum = np.sum(np.square(ary), 1)
    return - np.add(np.add(-2 * np.dot(ary, ary.T), ary_sum).T, ary_sum)


def softmax(ary: np.ndarray, diag_zero: bool = True, zero_index: Optional[int] = None):
    """
    Compute softmax values for each row of matrix X.
    (Implementation based around the following: https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/)
    """

    # Subtract max for numerical stability
    e_x = np.exp(ary - np.max(ary, axis=1).reshape([-1, 1]))

    # We usually want diagonal probabilities to be 0.
    if zero_index is None:
        if diag_zero:
            np.fill_diagonal(e_x, 0.)
    else:
        e_x[:, zero_index] = 0.

    # Add a tiny constant for stability of log we take later
    # e_x = e_x + 1e-8  # numerical stability
    e_x += 1e-8  # for numerical stability

    return e_x / e_x.sum(axis=1).reshape([-1, 1])


def calc_prob_matrix(distances: np.ndarray, sigmas: Optional[np.ndarray] = None, zero_index: Optional[int] = None):
    """
    Convert a distances matrix to a matrix of probabilities.
    (Implementation based around the following: https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/)
    """
    two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
    return softmax(distances / two_sig_sq, zero_index=zero_index)


def find_optimal_sigmas(distances: np.ndarray, target_perplexity: int, max_iterations: int = 10000) -> np.ndarray:
    """
    For each row of distances matrix, find sigma that results
    in target perplexity for that role.
    (Implementation based around the following: https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/)
    """
    sigmas = []
    # For each row of the matrix (each point in our dataset)
    for i in range(distances.shape[0]):
        # Binary search over sigmas to achieve target perplexity
        lower_binary_search = 1e-20
        upper_binary_search = 1000.
        for _ in range(max_iterations):
            guess = (lower_binary_search + upper_binary_search) / 2.
            prob_mat = calc_prob_matrix(distances[i:i+1, :], np.array(guess), i)
            val = 2 ** -np.sum(prob_mat * np.log2(prob_mat), 1)

            if val > target_perplexity:
                upper_binary_search = guess
            else:
                lower_binary_search = guess
            if np.abs(val - target_perplexity) <= 1e-10:
                break

        correct_sigma = guess

        # Append the resulting sigma to our output array
        sigmas.append(correct_sigma)
    return np.array(sigmas)


def get_tsne_plot(df: pd.DataFrame,
                  perplexity: int = 20,
                  random_seed: int = 0,
                  num_iterations: int = 1000,
                  learning_rate: int = 500,
                  color_col: str = 'serial_number_id',
                  color_discrete_map: Optional[dict] = None,
                  momentum: int = 0) -> go.Figure:
    """
    Estimates a SNE model, and plots the visualization of the given data.

    More information about t-SNE visualizations and how to use them effectively can be found here:
    https://distill.pub/2016/misread-tsne/

    (Implementation based around the following: https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/)

    :param df: A Pandas Dataframe of the the data to be visualized (e.g. a recording attribute DataFrame)
    :param perplexity: A value which can be thought of as determining how many neighbors of a point
     will be used to update its position in the visualization.
    :param random_seed: Integer value to set the seed value for the random
    :param num_iterations: The number of iterations to train for
    :param learning_rate: The rate at which to update teh values in the model
    :param color_col: The column name in the given dataframe (as merged_df) that is used to color
     data points with.   This is used in combination with the color_discrete_map parameter
    :param color_discrete_map: A dictionary which maps the values given to color data points based on (see the
     color_col parameter description) to the colors that these data points should be
    :param momentum:  The momentum to be used when applying updates to the model
    :return: Plotly figure of the, low-dimensional representation of the given data
    """
    rng = np.random.RandomState(random_seed)

    X = df.loc[:, np.float64 == df.dtypes].dropna(axis='columns').to_numpy()

    # Standardize the data
    X = (X - X.mean()) / X.std()

    # Obtain matrix of joint probabilities p_ij
    euc_distances = neg_squared_euc_dists(X)
    calc_prob_mat = calc_prob_matrix(
        euc_distances,
        find_optimal_sigmas(euc_distances, perplexity))

    P = (calc_prob_mat + calc_prob_mat.T) / (2. * calc_prob_mat.shape[0])

    # Initialise our 2D representation
    Y = rng.normal(0., 0.0001, [X.shape[0], 2])

    # Initialise past values (used for momentum)
    if momentum:
        Y_m2 = Y.copy()
        Y_m1 = Y.copy()

    # Start gradient descent loop
    for i in range(num_iterations):

        # Get Q and distances (distances only used for t-SNE)
        distances = neg_squared_euc_dists(Y)
        inv_distances = np.power(1. - distances, -1)
        np.fill_diagonal(inv_distances, 0.)
        q_distances = inv_distances / np.sum(inv_distances)

        # Estimate gradients with respect to Y
        y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
        grads = 4. * (np.expand_dims(P - q_distances, 2) * y_diffs * np.expand_dims(inv_distances, 2)).sum(1)

        # Update Y
        Y -= learning_rate * grads
        if momentum:  # Add momentum
            Y += momentum * (Y_m1 - Y_m2)
            # Update previous Y's for momentum
            Y_m2 = Y_m1.copy()
            Y_m1 = Y.copy()

    fig = px.scatter(
        x=Y[:, 0],
        y=Y[:, 1],
        color=df[color_col],
        color_discrete_map=color_discrete_map,
        hover_name=df.index,
    )

    fig.update_layout(
        xaxis_visible=False,
        yaxis_visible=False,
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
    )

    return fig
