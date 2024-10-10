import random
import numpy as np

def naive_neighborhood(i, n, N, data, random_seed):
    """
    Generate neighborhood by randomly change features, interval is determined by min/max of input data

    So far generate only ints (fine for ordinal variables, for float we can generate floats anyway)
    
    input
        i - instance position in data
        n - number of features to be changed
        N - number of neighbors to be generated
        data - data
        random_seed - random seed
    """

    random.seed(random_seed)

    x_max = data.max()
    x_min = data.min()
    
    # position of changed features
    inst_p = random.sample(range(0, len(x_max)), n)

    data_neigh = np.zeros((N, len(x_max)))

    for j in range(N):
        # original dp (Not scaled data)
        data_neigh[j, :] = data.iloc[i, :]    
        for k in range(n):
            # change data type (?!)
            data_neigh[j, inst_p[k]] = random.randint(x_min[inst_p[k]], x_max[inst_p[k]])

    return data_neigh

def naive_neighborhood_instance(instance_data, n, N, data, random_seed):
    """
    Generate neighborhood by randomly change features, interval is determined by min/max of input data

    So far generate only ints (fine for ordinal variables, for float we can generate floats anyway)
    
    input
        instance_data - full data of instance
        n - number of features to be changed
        N - number of neighbors to be generated
        data - data
        random_seed - random seed
    """

    random.seed(random_seed)
    
    print("data set size", np.shape(data))

    x_max = data.max(axis = 1)
    x_min = data.min(axis = 1)

    print("dimensions upper/lower bound arrays", len(x_max), len(x_min))
    
    # position of changed features
    inst_p = random.sample(range(0, len(x_max)), n)

    data_neigh = np.zeros((N, len(x_max)))

    for j in range(N):
        # original dp (Not scaled data)
        data_neigh[j, :] = instance_data    
        for k in range(n):
            if isinstance(x_min[inst_p[k]],float) == True:
                data_neigh[j, inst_p[k]] = random.uniform(x_min[inst_p[k]], x_max[inst_p[k]])
            else:
                # change data type (?!)
                data_neigh[j, inst_p[k]] = random.randint(x_min[inst_p[k]], x_max[inst_p[k]])

    return data_neigh
