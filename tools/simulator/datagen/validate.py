import csv
import os
import sys
import numpy as np
import ast


def mv_kullback_leibler_divergence(mean1, mean2, covar1, covar2, dep_columns):
    """
    Compute the KL-divergence for multi-variate distributions
    :param mean1: Mean vector of the first distribution
    :param mean2: Mean vector of the second distribution
    :param covar1: Covariance matrix of the first distribution
    :param covar2: Covariance matrix of the second distribution
    :param dep_columns: The list of linearly dependent columns
    :return:
    """
    ind_columns = []
    for i in range(0, mean1.size):
        if i not in dep_columns:
            ind_columns.append(i)

    mean1, covar1 = exclude_independent_columns(mean1, covar1, dep_columns)
    mean2, covar2 = exclude_independent_columns(mean2, covar2, dep_columns)
    return 0.5 * (
        np.matmul(np.linalg.inv(covar2), covar1).trace()
        + np.matmul(
            (mean2 - mean1).T, np.matmul(np.linalg.inv(covar2), mean2 - mean1)
        )
        - mean1.size
        + np.log(np.linalg.det(covar2) / np.linalg.det(covar1))
    )


def exclude_independent_columns(mean, covar, dep_columns):
    """
    Extract mean, covar only for the dependent columns
    :param mean: The mean vector
    :param covar: The covariance matrix
    :param dep_columns: The dependent columns to consider
    :return:
    """
    #
    nonzero_covar = []
    nonzero_covar_mean = []
    for c in range(0, covar.shape[0]):
        if c in dep_columns:
            row = []
            for v in range(0, covar[c].size):
                if v in dep_columns:
                    row.append(covar[c, v])
            nonzero_covar.append(row)
            nonzero_covar_mean.append(mean[c])

    return np.asarray(nonzero_covar_mean), np.array(nonzero_covar)


def get_distributions(input_path, int_columns):
    """
    Gather mean, standard deviation, and covariance
    :param input_path: The data set path to compute the distributions for
    :return:
    """
    with open(input_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        tuples = []
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            elif len(row) > 0:
                tup = []
                for i in range(0, len(row)):
                    tup.append(
                        int(row[i]) if i in int_columns else float(row[i])
                    )
                tuples.append(tup)
                line_count += 1
        data = (np.array(tuples).T).astype(np.float64)
        return np.mean(data, axis=1), np.cov(data)


def main():
    """
    Main method
    """
    # check inputs
    if len(sys.argv) < 2:
        print("Usage: Validate.py <distribution-dir> <datagen-dir>")
        sys.exit(1)

    # command line args
    dist_path = sys.argv[1]
    sim_path = sys.argv[2]

    # run the generator
    with open(os.path.join(dist_path, "distributions.csv")) as dist_file:
        dist_csv = csv.reader(dist_file, delimiter=",")

        validate_count = 0
        for ref_dist in dist_csv:

            if len(ref_dist) == 0:
                continue

            sim = os.path.join(sim_path, ref_dist[0])

            if not os.path.isfile(sim):
                continue

            mean1 = np.asarray(ast.literal_eval(ref_dist[1]))
            covar1 = np.asarray(ast.literal_eval(ref_dist[3]))
            dep_cols = np.asarray(ast.literal_eval(ref_dist[4]))
            int_cols = np.asarray(ast.literal_eval(ref_dist[5]))

            mean2, covar2 = get_distributions(sim, int_cols)

            try:
                klb = mv_kullback_leibler_divergence(
                    mean1, mean2, covar1, covar2, dep_cols
                )
                print("Group " + ref_dist[0] + ":", klb)
                validate_count += 1

            except Exception as e:
                print(e)

        print("\nValidated " + str(validate_count) + " jobs.\n")


main()
