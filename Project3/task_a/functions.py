import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as scpi
import seaborn as sns
import skimage.measure as skm
from numba import njit
from tqdm import tqdm


def dens_span_cluster(p, Lx, Ly, N):
    """
    Arguments:
        L: Integer giving length of box, given in number of unit cells
        p: Array containing various probabilities for a site to be occupied
        N: Monte-Carlo simulations
    Returns:
        P: Probability for a site to belong to a spanning cluster
    """
    nx  = len(p)
    P   = np.zeros(nx)

    for i in range(N):
        domain = np.random.rand(Lx, Ly)
        for j, pi in enumerate(p):
            binary_domain = domain < pi
            label, num = scpi.measurements.label(binary_domain)
            perc_x = np.intersect1d(label[0, :], label[-1, :])
            perc = perc_x[perc_x > 0]
            if len(perc) > 0:
                area = scpi.measurements.sum(binary_domain, label, perc[0])
                P[j] += area

    P /= (N * Lx * Ly)

    return P

def mass_cluster(p, Lx, Ly, N):
    """
    Arguments:
        L: Integer giving length of box, given in number of unit cells
        p: Float, probability for a site to be occupied
        N: Monte-Carlo simulations
    Returns:
        all_area: area of clusters, minus area of spanning clusters
    """
    all_area = []
    for i in range(N):
        domain = np.random.rand(Lx, Ly)
        binary_domain = domain < p
        label, num = scpi.measurements.label(binary_domain)
        label_list = np.arange(label.max() + 1)

        props = skm.regionprops(label)
        for prop in props:
            if not (prop.bbox[2] - prop.bbox[0] == Lx or
                    prop.bbox[3] - prop.bbox[1] == Ly):
                area = prop.area
                all_area.append(np.squeeze(area))
        # perc_x = np.intersect1d(label[0, :], label[-1, :])
        # percx = perc_x[np.where(perc_x == 0)]
        # perc_y = np.intersect1d(label[:, 0], label[:, -1])
        # percy = perc_y[np.where(perc_y == 0)]
        #
        # if len(percx) > 0:
        #     area = scpi.measurements.sum(binary_domain, label, percx[0])
        #     all_area.append(area)
        # elif len(percy) > 0:
        #     area = scpi.measurements.sum(binary_domain, label, percy[0])
        #     all_area.append(area)

    all_area = np.array(all_area)

    return all_area


def mass_spanning_cluster2(p, L, N):
    """
    Arguments:
        L: Integer giving length of box, given in number of unit cells
        p: Float, probability for a site to be occupied
        N: Monte-Carlo simulations
    Returns:
        all_area: area of spanning cluster
    """
    for i in range(N):
        domain = np.random.rand(L, L)
        binary_domain = domain < p
        label, num = scpi.measurements.label(binary_domain)
        label_list = np.arange(label.max() + 1)

        spanning = False

        props = skm.regionprops(label)
        for prop in props:
            if (prop.bbox[2] - prop.bbox[0] == L or
                prop.bbox[3] - prop.bbox[1] == L):
                area = prop.area
                spanning = True
                break

    if spanning:
        return area
    else:
        print('System not percolating')
        return 0


def mass_spanning_cluster(p, L, N):
    """
    Arguments:
        L: Array of integers giving length of box, given in number of unit cells
        p: Float, probability for a site to be occupied
        N: Monte-Carlo simulations
    Returns:
        all_area: area of spanning cluster
    """
    M = np.zeros(len(L))

    for i in range(N):
        for j, l in enumerate(L):
            domain = np.random.rand(l, l)
            binary_domain = domain < p
            label, num = scpi.measurements.label(binary_domain)

            perc_x = np.intersect1d(label[0, :], label[-1, :])
            percx = perc_x[np.where(perc_x > 0)]
            perc_y = np.intersect1d(label[:, 0], label[:, -1])
            percy = perc_y[np.where(perc_y > 0)]

            if len(percx) > 0:
                area = scpi.measurements.sum(binary_domain, label, percx[0])
                # all_area.append(area)
                M[j] += area
            elif len(percy) > 0:
                area = scpi.measurements.sum(binary_domain, label, percy[0])
                # all_area.append(area)
                M[j] += area
    M /= N

    return M


def mass_singly_con(p, L, N, walk):
    """
    Arguments:
        L: Array of integers giving length of box, given in number of unit cells
        p: Float, probability for a site to be occupied
        N: Monte-Carlo simulations
    Returns:
        M: mass of singly connected sites
    """
    M = np.zeros(len(L))

    for i in tqdm(range(N)):
        for j, l in enumerate(L):
            perc = []
            count = 0
            percolating = True
            while(len(perc) == 0):
                domain = np.random.rand(l, l)
                binary_domain = domain < p
                label, num = scpi.measurements.label(binary_domain)

                perc_x = np.intersect1d(label[0, :], label[-1, :])
                perc = perc_x[np.where(perc_x > 0)]

                count += 1
                if count > 1000:
                    print('No percolation')
                    percolating = False
                    break

            if percolating:
                labelList = np.arange(num + 1)
                binary_perc_domain = (label == perc[0])
                l, r = walk.walk(binary_perc_domain)

                sc_domain = l * r
                binary_sc_domain = sc_domain[np.where(sc_domain > 0)]
                M[j] += binary_sc_domain.shape[0]

            else:
                N

    M /= N

    return M


def dens_singly_con(p, L, N, walk):
    """
    Arguments:
        L: Integers giving length of box, given in number of unit cells
        p: Array of floats, probability for a site to be occupied
        N: Monte-Carlo simulations
    Returns:
        all_area: area of singly connected sites
    """
    P = np.zeros(len(p))

    for i in range(N):
        for j, pi in enumerate(p):
            perc = []
            c = 0
            percolating = True
            while(len(perc) == 0):
                domain = np.random.rand(L, L)
                binary_domain = domain < pi
                label, num = scpi.measurements.label(binary_domain)

                perc_x = np.intersect1d(label[0, :], label[-1, :])
                # perc_y = np.intersect1d(label[:, 0], label[:, -1])
                perc = perc_x[np.where(perc_x > 0)]
                # percy = perc_y[np.where(perc_y > 0)]

                # if len(percx) > 0:
                #     # print('percx')
                #     perc = percx
                # elif len(percy) > 0:
                #     perc = percy
                #     # print('percy')

                c += 1
                if c > 1000:
                    print('No percolation')
                    percolating = False
                    break

            if percolating:
                binary_perc_domain = (label == perc[0])
                l, r = walk.walk(binary_perc_domain)

                sc_domain = l * r
                binary_sc_domain = sc_domain[np.where(sc_domain > 0)]
                P[j] += binary_sc_domain.shape[0]

    P /= N * L * L

    return P


def DensSpanClusterAltGeo(p, Lx, Ly, N):
    nx  = len(p)
    P   = np.zeros(nx)

    domain = np.random.rand(Lx, Ly)
    domain[:, :Ly//2] = None  # Forcing the values to be one = unoccupied

    for j, pi in enumerate(p):
        binary_domain = domain < pi
        label, num = scpi.measurements.label(binary_domain)
        perc_x = np.intersect1d(label[0, :], label[-1, :])
        perc = perc_x[perc_x > 0]
        if len(perc) > 0:
            area = scpi.measurements.sum(binary_domain, label, perc[0])
            P[j] += area

    P /= (N * Lx * Ly)

    return P


def main():
    N_p = 100    # Number of probailities p
    p_min = 0.1
    p_max = 1.0

    Lx = 100  # Size of system
    Ly = Lx

    N = 1000

    p = np.linspace(p_min, p_max, N_p)

    P = DensSpanCluster(p, Lx, Ly, N)

    plot_P(p, P, L)


if __name__ == "__main__":
    sns.set()
    np.random.seed(42)

    main()
