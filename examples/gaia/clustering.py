import argparse
import os
import warnings
from itertools import chain

import numpy as np
import pandas as pd
from pycompss.api.api import compss_barrier
from pycompss.api.task import task

from dislib.data import Subset, Dataset
from dislib.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
import time

header = ['l', 'b', 'par', 'pmra', 'pmdec']

lMinim = 0.
lMaxim = 360.
bMinim = -20.
bMaxim = 20.

box_sizes = [12, 13, 14, 15, 16]
box_sizes.reverse()
min_samples = [5, 6, 7, 8, 9]
min_samples.reverse()

tol = 0.2

s_time = time.time()

# In case we want to do a hierarchical model (1 means apply clustering once)
repetitions = 1

n_simulations = 30


def _do_clustering(df_clean, part_size, n_regions, out_dir):
    params = [(L, minPts) for L in box_sizes for minPts in min_samples]
    db_results_by_param = []
    for L, minPts in params:
        db_results = _compute_boxes(df_clean, L, minPts, n_regions,
                                    part_size)
        db_results_by_param.append(db_results)

    for L, minPts in params:
        db_results = db_results_by_param.pop(0)

        _postprocess_results(out_dir, L, minPts, *chain(*db_results))

    print("Task Creation Time: ", "{0:.2f}".format(time.time() - s_time))
    compss_barrier()
    print("Main Execution Time: ", "{0:.2f}".format(time.time() - s_time))


@task()
def _postprocess_results(out_dir, L, minPts, *db_results):
    # Work with data at every box
    columns = header + ['cluster_label', 'cluster_core']
    all_clusters = pd.DataFrame(columns=columns, dtype=float)
    n_label = 0

    for db_res in zip(*[iter(db_results)] * 11):
        shift, i, j, lMin, Ll, bMin, Lb, df_box, dens, components, res_subset = db_res
        labels = res_subset.labels
        n_clusters = len(components)

        for c in range(n_clusters):
            # find members and cores
            members = _get_members(df_box, c, labels)

            # check if there are members in the edge
            edge = _in_edge(df_box, members, lMin, Ll, bMin, Lb, i, j)

            # check if we have already counted the cluster
            counted = _counted_cluster(df_box, all_clusters, members)

            if not edge and not counted:
                df_members = df_box.loc[members].assign(
                    cluster_label=n_label).assign(dens=dens)
                all_clusters = all_clusters.append(df_members)
                n_label = n_label + 1

    # save a dataframe with all the found clusters
    name = "all_clusters_L{}_minPts{}.csv".format(L, minPts)
    columns = header + ['cluster_label', 'dens']
    all_clusters[columns].to_csv(os.path.join(out_dir, name))


def _compute_boxes(df_clean, L, minPts, n_regions, part_size):
    k_neighbors = minPts - 1

    # Compute number of boxes and actual size
    nl = int(np.floor((lMaxim - lMinim) / L))
    nb = int(np.floor((bMaxim - bMinim) / L))
    Ll = (lMaxim - lMinim) / nl
    Lb = (bMaxim - bMinim) / nb

    shifts = [0., 1./3., 2./3.]
    db_results = []

    for reps in range(repetitions):
        for shift in shifts:
            db_results = _compute_shift(df_clean, shift, Lb, nb, Ll, nl, db_results,
                                        k_neighbors, minPts, part_size,
                                        n_regions)
    return db_results


def _compute_shift(df_clean, shift, Lb, nb, Ll, nl, db_results, k_neighbors,
                   minPts, part_size, n_regions):
    lMin = lMinim - shift * Ll
    bMin = bMinim - shift * Lb
    if shift == 0.:
        nll = nl
        nbb = nb
    else:
        # If the boxes are shifted, add one box to cover all the region
        nll = nl + 1
        nbb = nb + 1

    clean_query = '@lMin + @i*@Ll < l < @lMin + (@i+1)*@Ll and @bMin + @j*@Lb < b < @bMin + (@j+1)*@Lb'

    print("Computing ", nll * nbb, " regions. Time: ",
          "{0:.2f}".format(time.time() - s_time), flush=True)

    for i in range(nll):
        for j in range(nbb):

            # Count stars in actual box
            l = df_clean['l'].values
            b = df_clean['b'].values
            samples_size = np.count_nonzero((lMin + i*Ll < l) & (l < lMin + (i+1)*Ll) & (bMin + j*Lb < b) & (b < bMin + (j+1)*Lb))
            if samples_size <= k_neighbors:
                warnings.warn("Samples size is lower than k_neighbors")
                pass
            else:
                df_box, subset, dens, epsilon = _preprocess_box(df_clean, k_neighbors, Ll, Lb, lMin, bMin, i, j)
                dataset = Dataset(len(header))
                dataset.append(subset, samples_size)

                # region_widths = (df_scaled.max() - df_scaled.min()) / n_regions
                # check_valid_epsilon(epsilon, region_widths)

                res, db = _run_dbscan(dataset, epsilon, minPts,
                                      n_regions)
                res_subset = res[0]
                components = db._components
                db_results.append((shift, i, j, lMin, Ll, bMin, Lb, df_box, dens, components, res_subset))

    return db_results


@task(returns=4)
def _preprocess_box(df_clean, k_neighbors, Ll, Lb, lMin, bMin, i, j):
    df_box = df_clean.query('@lMin + @i*@Ll < l < @lMin + (@i+1)*@Ll and @bMin + @j*@Lb < b < @bMin + (@j+1)*@Lb')
    samples_size = len(df_box)
    dens = _get_density(df_box, samples_size)
    epsilon = _compute_epsilon(df_box, k_neighbors, samples_size)
    data_scaled = StandardScaler().fit_transform(
        np.array(df_box[header]))
    df_scaled = pd.DataFrame(data=data_scaled,
                             columns=df_box.columns, dtype=float)
    subset = Subset(np.array(df_scaled))
    return df_box, subset, dens, epsilon


def _counted_cluster(df_box, all_clusters, members):
    return (any(df_box.loc[members, 'l'].isin(all_clusters['l']))) or (
        any(df_box.loc[members, 'b'].isin(all_clusters['b'])))


def _in_edge(df_box, members, lMin, Ll, bMin, Lb, i, j):
    # check if any member falls in the edge.
    minl = min(df_box.loc[members, 'l'])
    maxl = max(df_box.loc[members, 'l'])
    minb = min(df_box.loc[members, 'b'])
    maxb = max(df_box.loc[members, 'b'])

    return minl < (lMin + i * Ll + tol) or maxl > (
            lMin + (i + 1) * Ll - tol) or minb < (
                   bMin + j * Lb + tol) or maxb > (
                   bMin + (j + 1) * Lb - tol)


def _get_members(df_box, c, labels):
    return df_box.iloc[[t for (t, lab) in enumerate(labels) if lab == c]].index


def _get_density(df_box, samples_size):
    l_max = df_box['l'].max()
    l_min = df_box['l'].min()
    b_max = df_box['b'].max()
    b_min = df_box['b'].min()

    return float(samples_size) / ((l_max - l_min) * (b_max - b_min))


def _run_dbscan(dataset, epsilon, minPts, n_regions):
    db = DBSCAN(eps=epsilon, min_samples=minPts, n_regions=n_regions, dimensions=[0,1], arrange_data=False)
    db.fit(dataset)
    # return dataset.labels, db.n_clusters
    return dataset, db


# Computation of epsilon (parameter needed for DBSCAN)
def _compute_epsilon(df_box, k, N):
    """
    Compute epsilon for a given

    Input
        df_box: Non-Standarised DataFrame where to compute epsilon
        k: number of nearest neighbour to take into account in the computation

    Output
        espilon
    """
    # Standarisation of features (5-D)
    data_scaled = StandardScaler().fit_transform(
        np.array(df_box[header]))
    df_scaled = pd.DataFrame(data=data_scaled, columns=df_box.columns,
                             dtype=float)
    # Compute kth nearest neighbour distance to each source
    neighs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(
        np.array(df_scaled))
    dist, ind = neighs.kneighbors(df_scaled)
    kDist = [dist[t, k - 1] for t in range(len(dist))]
    # Here we want to draw a sample of only field stars (no clusters)
    min_kDist_sim = []
    # Repeat the process 30 times to reduce random effects
    for s in range(30):
        m = []
        for (s, par) in enumerate(header):
            # Capture the distribution of the field
            kernel = gaussian_kde(df_box[par])
            # Random resampling from the captured distribution (as it is a random resampling, no clusters are expected)
            resamp = kernel.resample(N)[0]
            m.append(resamp)
        # Standarise the features the same way we did for the real data
        df_sim = pd.DataFrame(data=np.matrix(m).T, columns=df_box.columns,
                              dtype=float)
        data_sim_scaled = StandardScaler().fit_transform(
            np.array(df_sim[header]))
        df_sim_scaled = pd.DataFrame(data=data_sim_scaled,
                                     columns=df_sim.columns, dtype=float)
        neighs_sim = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(
            np.array(df_sim_scaled))  # scaled
        dist_sim, ind_sim = neighs_sim.kneighbors(df_sim_scaled)  # scaled
        kDist_sim = [dist_sim[t, k - 1] for t in range(len(dist_sim))]
        min_kDist_sim.append(np.min(kDist_sim))
    e_max = np.mean(min_kDist_sim)
    # Take the mean between the minimum of the (real data) kth nearest neighbour distribution (where we expect low values corresponding to the data) and the minimum of (simulated) field stars
    epsilon = (1. / 2.) * (np.min(kDist) + e_max)
    return epsilon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", metavar="N_REGIONS", type=int, required=True)
    parser.add_argument("-o", metavar="OUT_LOCATION", type=str, required=True)
    parser.add_argument("-p", metavar="PART_SIZE", type=int, required=True)
    parser.add_argument("input_data", type=str)
    args = parser.parse_args()


    df = pd.read_csv(args.input_data)[header]
    query = "0. < par < 7. and -30. < pmra < 30. and -30. < pmdec < 30."
    df_clean = df.query(query)[header]
    if not os.path.exists(args.o):
        os.mkdir(args.o)
    _do_clustering(df_clean, args.p, args.r, args.o)


if __name__ == "__main__":
    main()
