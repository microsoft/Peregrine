"""Module for generating synthetic data set based on the given data distribution
"""
import sys
import math
import numpy as np
import csv
import os


def gaussian(N, d):
    """
    Generate normal distribution using Box-Muller transform.
    :param N: number of instances to generate
    :param d: the dimension of each instance
    :return:
    """
    columns = []
    for i in range(0, int((d + 1) / 2)):
        u1 = np.random.rand(N)
        u2 = np.random.rand(N)
        z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        columns.append(z1)
        if len(columns) < d:
            z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
            columns.append(z2)
    return np.array(columns)


def generate_dependent_data(columns, mean, covar, N):
    """
    Generate correlated data
    https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
    :param columns: The list of dependent columns
    :param mean: The mean vector of the columns
    :param covar: The covariance matrix
    :param N: The number of instances to generate
    :return:
    """
    # extract mean, covar only for the dependent columns
    nonzero_covar = []
    nonzero_covar_mean = []
    for c in range(0, covar.shape[0]):
        if c in columns:
            row = []
            for v in range(0, covar[c].size):
                if v in columns:
                    row.append(covar[c, v])
            nonzero_covar.append(row)
            nonzero_covar_mean.append(mean[c])
    nonzero_covar = np.array(nonzero_covar)
    nonzero_covar_mean = np.asarray(nonzero_covar_mean)

    # factorize
    A = np.linalg.cholesky(nonzero_covar)  # Cholesky decomposition

    # generate normal distribution data
    z = gaussian(N, nonzero_covar_mean.size)

    # adjust the distribution
    return nonzero_covar_mean[:, np.newaxis] + np.matmul(A, z)


def generate_independent_data(columns, mean, std, N):
    """
    Generate independent variables
    :param columns:
    :param mean:
    :param std:
    :param N:
    :return:
    """
    # generate normal distribution data
    z = gaussian(N, len(columns))

    # adjust with mean and standard deviation
    data = []
    for i in range(0, len(columns)):
        items = []
        for v in z[i]:
            items.append(mean[columns[i]] + std[columns[i]] * v)
        data.append(items)

    # return in matrix form
    return np.array(data)


def write_output(
    dep_cols,
    dep_data,
    ind_cols,
    ind_data,
    sim_output,
    consolidated_csv,
    hash_tag,
    header,
    int_cols,
):
    """
    Write output
    :param dep_cols: dependent columns
    :param dep_data: data for dependent columns
    :param ind_cols: independent columns
    :param ind_data: data for independent columns
    :param sim_output: the simulated output path
    :param consolidated_csv: the consolidated output path
    :param hash_tag: the hash tag of the job whose datagen is being written to the output
    :param header: the file header
    :param int_cols: the list of integer columns
    """
    with open(sim_output, "w", newline="\n") as myfile:
        wr = csv.writer(myfile)
        wr.writerow(header)

        ddata = dep_data.T
        idata = ind_data.T
        if (
            len(ddata) != len(idata)
            and ddata.shape[0] != 0
            and idata.shape[0] != 0
        ):
            print("Dependent data shape", ddata.shape)
            print("Independent data shape", idata.shape)
            raise Exception(
                "Different #rows in dependent and independent columns"
            )

        for r in range(0, len(ddata)):
            d_idx = 0
            i_idx = 0
            row = []
            for i in range(0, len(header)):
                if i in dep_cols:
                    value = (
                        math.ceil(ddata[r, d_idx])
                        if i in int_cols
                        else ddata[r, d_idx]
                    )
                    row.append(abs(value))
                    d_idx += 1
                elif i in ind_cols:
                    value = (
                        math.ceil(idata[r, i_idx])
                        if i in int_cols
                        else idata[r, i_idx]
                    )
                    row.append(abs(value))
                    i_idx += 1
                else:
                    raise Exception(
                        "column neither in dependent nor in independent column list",
                        i,
                    )
            wr.writerow(row)
            row.insert(0, hash_tag)
            consolidated_csv.writerow(row)


def generate(input_dist, sim_output, consolidated_csv, hash_tag, first, size):
    """
    Generate simulated data set
    :param input_dist: The base path for data distribution
    :param sim_output: The output path for simulated data set
    :param consolidated_csv: The writer for consolidated CSV
    :param hash_tag: The hash tag for this recurring job
    :param first: Flag to indicate whether this is the first recurring job
    :param size: The number of instances to simulate
    """
    header = None
    with open(input_dist + ".header") as my_file:
        for row in csv.reader(my_file, delimiter=","):
            header = row
            break

    if first:
        h = header.copy()
        h.insert(0, "HT1")
        consolidated_csv.writerow(h)

    mean = np.load(input_dist + ".mean.npy")
    stdev = np.load(input_dist + ".stdev.npy")
    covar = np.load(input_dist + ".covar.npy")
    dep_columns = np.load(input_dist + ".depcols.npy").tolist()
    int_columns = np.load(input_dist + ".intcols.npy").tolist()

    # separate dependent and independent columns
    ind_columns = []
    for i in range(0, mean.size):
        if i not in dep_columns:
            ind_columns.append(i)

    # generate dependent and independent column data
    dep_data = generate_dependent_data(dep_columns, mean, covar, size)
    ind_data = generate_independent_data(ind_columns, mean, stdev, size)

    # write to file
    write_output(
        dep_columns,
        dep_data,
        ind_columns,
        ind_data,
        sim_output,
        consolidated_csv,
        hash_tag,
        header,
        int_columns,
    )


def main():
    """
    Main method
    """
    # check inputs
    if len(sys.argv) < 4:
        print(
            "Usage: simulate.py <distribution-dir> <datagen-dir> <consolidated_output> <size-per-query>"
        )
        sys.exit(1)

    # command line args
    dist_path = sys.argv[1]
    sim_path = sys.argv[2]
    consolidated_output_path = sys.argv[3]
    size_per_query = int(sys.argv[4])

    if not os.path.exists(sim_path):
        os.mkdir(sim_path)

    first = True

    # run the generator
    with open(os.path.join(dist_path, "meta")) as csv_file, open(
        consolidated_output_path, "w", newline="\n"
    ) as consolidated_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        consolidated_csv = csv.writer(consolidated_file)

        success_count = 0
        for data in csv_reader:

            if len(data) == 0:
                continue

            input_dist = os.path.join(dist_path, data[0])
            sim_output = os.path.join(sim_path, data[0])

            try:
                generate(
                    input_dist,
                    sim_output,
                    consolidated_csv,
                    data[0],
                    first,
                    size_per_query,
                )
                success_count += 1
            except np.linalg.LinAlgError as e:
                print("Failed to generate data set for " + data[0] + ".", e)
            first = False

        print("\nSuccessfully simulated " + str(success_count) + " jobs.\n")


main()
