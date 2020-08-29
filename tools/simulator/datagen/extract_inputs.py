"""Module for extracting data set characteristics.
"""
import sys
import csv
import os
from numpy import std, array, mean, save, cov
import numpy as np
import json


def group_inputs(
    input_file,
    group_key,
    hash_tag,
    columns,
    output_path,
    max_groups=10,
    support_threshold=5,
):
    """Parse and group the AutoToken input into different recurring pipelines
    :param input_file: The data set path to parse
    :type input_file: string
    :param group_key: The index of the grouping key
    :type group_key: int
    :param hash_tag: The index of the job's recurring hash tag
    :type hash_tag: int
    :param columns: The list of columns to extract from the data set
    :type columns: list
    :param output_path: The output path to store the grouped data sets
    :type output_path: string
    :param max_groups: The maximum number of groups to create
    :type max_groups: int, optional
    :param support_threshold: The minimum support for each group
    :type support_threshold: int, optional
    :return: list of group data sets, list of group hash values
    """
    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        group_tuples = {}
        group_hash_tag = {}
        header = []
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                for col in columns:
                    header.append(row[col])
            elif len(row) > 0:
                if not row[group_key] in group_tuples:
                    group_tuples[row[group_key]] = []
                    group_hash_tag[row[group_key]] = row[hash_tag]
                tup = []
                for col in columns:
                    tup.append(float(row[col]))
                group_tuples[row[group_key]].append(tup)
                line_count += 1

            if line_count % 100000 == 0:
                print(f"Processed {line_count} lines.")

        files, group_hashes = [], []
        for group in group_tuples:

            if len(group_tuples[group]) < support_threshold:
                # print("Too few instances for", group)
                continue
            try:
                file = os.path.join(output_path, group_hash_tag[group])
                with open(file, "w") as my_file:
                    writer = csv.writer(my_file)
                    writer.writerow(header)
                    for tup in group_tuples[group]:
                        writer.writerow(tup)

                files.append(file)
                group_hashes.append(group_hash_tag[group])
                if len(files) >= max_groups:
                    break

            except Exception as ex:
                print("Failed to write group:" + group, ex)

        return files, group_hashes


def store_distributions(
    output_path, data_files, group_hashes, int_columns, int_columns_shifted
):
    """
    Gather and collect the distributions from each of the data files separately.
    :param output_path: The output path to store the distributions
    :type output_path: string
    :param data_files: The list of grouped data sets to compute the distributions on
    :type data_files: list
    :param group_hashes: The list of hash values used for grouping the data sets
    :type group_hashes: list
    :param int_columns: The list of integer columns in the input file
    :param int_columns_shifted: The list of integer columns in the output file
    :type int_columns: list
    """
    header = None
    with open(
        os.path.join(output_path, "distributions.csv"), "w"
    ) as dist_file:
        dist_writer = csv.writer(dist_file)
        for (data_file, group_hash) in zip(data_files, group_hashes):
            header, mean, stdev, covar, dep_cols = get_distributions(
                data_file, int_columns
            )
            row = [
                group_hash,
                mean.tolist(),
                stdev.tolist(),
                covar.tolist(),
                dep_cols.tolist(),
                int_columns_shifted,
            ]
            dist_writer.writerow(row)
    with open(os.path.join(output_path, "header.csv"), "w") as header_file:
        header_writer = csv.writer(header_file)
        header_writer.writerow(header)


def get_distributions(data_set, int_columns):
    """
    Gather mean, standard deviation, and co-variation
    :param data_set: The input to get the distributions from
    :type data_set: string
    :param int_columns: The list of integer columns
    :type int_columns: list
    :return: header, mean. standard deviation, co-variation, dependent columns
    """
    header = None
    with open(data_set) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        tuples = []
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                header = row
            elif len(row) > 0:
                tup = []
                for i in range(0, len(row)):
                    value = float(row[i])
                    tup.append(int(value) if i in int_columns else value)
                tuples.append(tup)
                line_count += 1
        data = (array(tuples).T).astype(np.float64)

        covar = cov(data)
        dep_cols = get_dependent_columns(covar)
        return (
            header,
            mean(data, axis=1),
            std(data, axis=1),
            covar,
            np.array(dep_cols),
        )


def get_dependent_columns(covar):
    """
    Get the list of dependent columns
    :param covar: The covariance matrix
    :return: Dependent columns
    """
    ind_columns = (np.where(~covar.any(axis=1))[0]).tolist()
    dep_columns_z = []
    for i in range(0, covar.shape[0]):
        if i not in ind_columns:
            dep_columns_z.append(i)
    return exclude_linear_combination_variables(covar, dep_columns_z)


def exclude_linear_combination_variables(covar, all_columns):
    """
    Exclude columns that are linear combination of other columns in the covariance matrix.
    :param covar: The covariance matrix
    :param all_columns: The set of all columns
    :return: List of dependent columns
    """
    dep_columns = [all_columns[0]]
    for i in range(1, len(all_columns)):
        columns = dep_columns.copy()
        columns.append(all_columns[i])
        test_covar = []
        for c in range(0, covar.shape[0]):
            if c in columns:
                row = []
                for v in range(0, covar[c].size):
                    if v in columns:
                        row.append(covar[c, v])
                test_covar.append(row)
        test_covar = np.array(test_covar)
        if np.linalg.matrix_rank(test_covar) == test_covar.shape[0]:
            dep_columns.append(all_columns[i])
    return dep_columns


def main():
    """
    Main method
    """
    # check inputs
    if len(sys.argv) < 6:
        print(
            "Usage: extract.py <autotoken-training-input> <extraction-dir> "
            "<distribution-dir> <max-queries> <support-threshold>"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    extract_path = sys.argv[2]
    dist_path = sys.argv[3]
    max_queries = int(sys.argv[4])
    support_threshold = int(sys.argv[5])

    # These constants depend on the training input schema
    # if the training input schema changes, then these have to change as well!
    hash_tag = 2
    columns = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    int_columns = [1, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16]

    if len(sys.argv) == 7:
        group_key = int(sys.argv[6])
    else:
        group_key = hash_tag  # default group key is HT1

    # update positions to correspond to column numbers in the per group-key
    # generated datasets. Gaps in sequence will be removed. Also, accounts
    # for the fact that group_key will not be explicitly included in the
    # generated data per group_key value
    pos_map = {}
    j = 0
    for i in columns:
        pos_map[i] = j
        j += 1
    int_columns_shifted = []
    for i in int_columns:
        int_columns_shifted.append(pos_map[i])

    if not os.path.exists(extract_path):
        os.mkdir(extract_path)
    files, group_hashes = group_inputs(
        input_file,
        group_key,
        hash_tag,
        columns,
        extract_path,
        max_queries,
        support_threshold,
    )
    print("\nSuccessfully extracted " + str(len(files)) + " jobs.")

    if not os.path.exists(dist_path):
        os.mkdir(dist_path)
    store_distributions(
        dist_path, files, group_hashes, int_columns, int_columns_shifted
    )
    print("Stored job distributions.\n")


main()
