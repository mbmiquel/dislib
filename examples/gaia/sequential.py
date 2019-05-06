# Import packages
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import time

t0 = time.time()


# Define functions that will perform the task
# Computation of epsilon (parameter needed for DBSCAN)
def compute_epsilon(df_box, k, N):
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
        np.array(df_box[['l', 'b', 'par', 'pmra', 'pmdec']]))
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
        for (s, par) in enumerate(['l', 'b', 'par', 'pmra', 'pmdec']):
            # Capture the distribution of the field
            kernel = gaussian_kde(df_box[par])
            # Random resampling from the captured distribution (as it is a random resampling, no clusters are expected)
            resamp = kernel.resample(N)[0]
            m.append(resamp)
        # Standarise the features the same way we did for the real data
        df_sim = pd.DataFrame(data=np.matrix(m).T, columns=df_box.columns,
                              dtype=float)
        data_sim_scaled = StandardScaler().fit_transform(
            np.array(df_sim[['l', 'b', 'par', 'pmra', 'pmdec']]))
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


# DBSCAN
# def run_dbscan(df_box,epsilon,minPts,all_clusters,n_label,dens,tol = 0.2):
def run_dbscan(df_box, epsilon, minPts, dens, lMin, bMin, i, j, Ll, Lb, nbox,
               tol=0.2):
    """
    Run DBSCAN and return the clusters found

    Input
        df_box: Non-Standarised DataFrame where to crun DBSCAN
        epsilon: Computed parameter needed for DBSCAN
        minPts: Minimum size of the clusters to find
        all_clusters: DataFrame with the already found clusters. To see if the newly found are already detected
        tol: thickness of the edges of the region, no cluster has to fall in the edges. Tolerance when considering a cluster in the edge of the box (in degrees)

    Output
        found_clusters: DataFrame containing all found clusters, with member stars
    """
    n_label = 0
    # Standarisation of features (5-D)
    data_scaled = StandardScaler().fit_transform(
        np.array(df_box[['l', 'b', 'par', 'pmra', 'pmdec']]))
    df_scaled = pd.DataFrame(data=data_scaled, columns=df_box.columns,
                             dtype=float)
    # Create empty dataframe to include all found clusters
    found_clusters = pd.DataFrame(columns=list(
        ['l', 'b', 'par', 'pmra', 'pmdec', 'cluster_label', 'cluster_core']),
                                  dtype=float)

    print(epsilon, minPts)

    # Run DBSCAN with the previously computed epsilon and the parameter minPts
    db = DBSCAN(eps=epsilon, min_samples=minPts).fit(np.array(df_scaled))
    # Number of clusters found
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

    print("Time: ",
          "{0:.2f}".format(time.time() - t0),
          ". Found ", n_clusters, " clusters.",
          flush=True)
    # For each cluster, find the just members or cores.
    for c in range(n_clusters):
        members = df_box.iloc[
            [t for (t, lab) in enumerate(db.labels_) if lab == c]].index
        cores = df_box.iloc[db.core_sample_indices_].index
        minl = min(df_box.loc[members, 'l'])
        maxl = max(df_box.loc[members, 'l'])
        minb = min(df_box.loc[members, 'b'])
        maxb = max(df_box.loc[members, 'b'])
        # See if any of the members fall in the edge of the box
        if minl < (lMin + i * Ll + tol) or maxl > (lMin + (i + 1) * Ll - tol) \
                or minb < (bMin + j * Lb + tol) or maxb > (
                bMin + (j + 1) * Lb - tol):
            pass
        else:
            # Check if the cluster was already counted in previous shifts
            # if (not any(df_box.loc[members,'l'].isin(all_clusters['l']))) and \
            # (not any(df_box.loc[members,'b'].isin(all_clusters['b']))):
            df_members = df_box.loc[members].assign(
                cluster_label=n_label).assign(dens=dens).assign(box=nbox)
            found_clusters = found_clusters.append(df_members)
            n_label = n_label + 1
            for memb in members:
                # Label the cores of the clusters
                if memb in cores:
                    found_clusters.at[memb, 'cluster_core'] = 1
                else:
                    found_clusters.at[memb, 'cluster_core'] = 0
    return found_clusters


# Function that computes epsilon and runs dbscan. Because both functions work on the same dataframe (box), is it preferable to parallelise this function?
# def computation_in_each_box(df_box,minPts,all_clusters,n_label):
def computation_in_each_box(df_box, minPts, lMin, bMin, i, j, Ll, Lb, nbox):
    """
    Higher level function that performs the whole analysis in each box

    Input
        df_box: Non-standarised DataFrame of the region to study
        minPts: Minimum size of the clusters to find

    Output
        found_clusters: DataFrame containing all found clusters, with member stars
    """
    # Parameter for the calculation of epsilon (related to the size of the clusters). There will be different epsilons for different divisions of the sky
    k = minPts - 1
    # Number of sources in the box
    N = len(df_box)
    # Compute density of the box (use it as a flag for the post-analysis)
    dens = float(N) / ((df_box['l'].max() - df_box['l'].min()) * (
                df_box['b'].max() - df_box['b'].min()))
    epsilon = compute_epsilon(df_box, k, N)
    # found_clusters = run_dbscan(df_box,epsilon,minPts,all_clusters,n_label,dens)
    found_clusters = run_dbscan(df_box, epsilon, minPts, dens, lMin, bMin, i,
                                j, Ll, Lb, nbox)
    return found_clusters


def main():
    # Read data file
    df = \
    pd.read_csv("df_tgas_real.csv")[
        ['l', 'b', 'par', 'pmra', 'pmdec']]
    #df = df.sample(frac=0.1)
    
    if not os.path.exists('out'):
        os.mkdir('out')

    # Clean data of outliers
    df_clean = \
    df.query("0. < par < 7. and -30. < pmra < 30. and -30. < pmdec < 30.")[
        ['l', 'b', 'par', 'pmra', 'pmdec']]

    # Define parameters of the sky region, typical size of boxes that divides the sky, and the minimum points to consider a cluster
    lMinim = 0.  # limits of sky region to study
    lMaxim = 360.
    bMinim = -20.
    bMaxim = 20.

    # Loop over the parameters of DBSCAN. This sets the number and size of the divisions on the sky (L), and the sizes of the clusters to find (minPts)
    for L in [12, 13, 14, 15, 16]:
        for minPts in [5, 6, 7, 8, 9]:
            # Compute number and size of divisions of the sky. This depends only on the limits of the region to study and the L parameter
            nl = int(np.floor((lMaxim - lMinim) / L))
            nb = int(np.floor((bMaxim - bMinim) / L))
            Ll = (lMaxim - lMinim) / nl
            Lb = (bMaxim - bMinim) / nb

            # Create a new (empty) dataframe to list all the found clusters
            # all_clusters = pd.DataFrame(columns = list(['l','b','par','pmra','pmdec','cluster_label','cluster_core']),dtype = float)
            clusters_list = []
            # n_label = 0

            # flag for box number (for future cluster label)
            nbox = 0

            # Shift the grid of boxes in the sky to account for clusters in the border
            for shift in [0., 1. / 3., 2. / 3.]:
                lMin = lMinim - shift * Ll
                bMin = bMinim - shift * Lb
                # Add extra box when they are shifted to cover all the region to study
                if shift == 0.:
                    nll = nl
                    nbb = nb
                else:
                    nll = nl + 1
                    nbb = nb + 1
                # Loop for all boxes in sky. This is what we should parallelise first. Each box should go to a different task, so we won't compute the boxes one after the other.
                # The only input of each task would be the dataframe of the given box.
                for i in range(nll):
                    for j in range(nbb):
                        # Select stars that are in the actual box
                        df_box = df_clean.query(
                            '@lMin + @i*@Ll < l < @lMin + (@i+1)*@Ll and @bMin + @j*@Lb < b < @bMin + (@j+1)*@Lb')
                        # found_clusters = computation_in_each_box(df_box,minPts,all_clusters,n_label)
                        found_clusters = computation_in_each_box(df_box,
                                                                 minPts, lMin,
                                                                 bMin, i, j,
                                                                 Ll, Lb, nbox)
                        clusters_list.append(found_clusters)
                        nbox = nbox + 1

            all_clusters = pd.DataFrame(columns=list(
                ['l', 'b', 'par', 'pmra', 'pmdec', 'cluster_label',
                 'cluster_core']), dtype=float)
            all_clusters = all_clusters.append(clusters_list)

            new_label = 0
            all_clusters_non_repeated = pd.DataFrame(
                columns=all_clusters.columns)
            for i in np.unique(np.array(all_clusters['box'])):
                box = all_clusters.query('box == @i')
                for j in np.unique(np.array(box['cluster_label'])):
                    clus = box.query('cluster_label == @j')
                    if (not any(clus['l'].isin(
                            all_clusters_non_repeated['l']))) and (
                    not any(clus['b'].isin(all_clusters_non_repeated['b']))):
                        clus['new_label'] = new_label
                        all_clusters_non_repeated = all_clusters_non_repeated.append(
                            clus)
                        new_label = new_label + 1
            all_clusters_non_repeated = all_clusters_non_repeated.drop(
                ['cluster_label'], axis=1)
            all_clusters_non_repeated = all_clusters_non_repeated.rename(
                index=str, columns={'new_label': 'cluster_label'})

            # save a dataframe with all the found clusters
            name = "out/sequential_clusters_L{}_minPts{}.csv".format(L, minPts)
            all_clusters_non_repeated[
                ['l', 'b', 'par', 'pmra', 'pmdec', 'cluster_label',
                 'cluster_core', 'dens']].to_csv(name)
    print(time.time() - t0)


if __name__ == "__main__":
    main()