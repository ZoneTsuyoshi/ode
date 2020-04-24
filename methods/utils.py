import numpy as np


def nondelay_embedding(x, E=3, n=1, seed=1):
    """
    Args:
        x [n_timesteps, n_dim]: data
        E {int}: embedding dimension
        n {int}: number of embeddings for generating

    Return:
        [n, n_timesteps, E]
    """
    np.random.seed(seed)
    assert x.ndim==2
    _, n_dim = x.shape

    combinations = np.random.randint(n_dim, size=(n,E))
    return x[:, combinations].swapaxes(0,1)


def delay_embedding(x, E=None, max_delay=5, n=1, seed=1):
    """
    Args:
        x [n_timesteps]: data
        E {int}: embedding dimension
        max_delay {int}: maximum embedding dimension
        n {int}: number of embeddings for generating

    Return:
        [n, n_timesteps - delay, E] or [n_timesteps - delay, E]
    """
    np.random.seed(seed)
    n_timesteps = len(x)

    # only maximum number of delay is determined
    if E is None:
        # binary array for embedding
        # if True, the correspoding delay interval is used.
        results = np.zeros((n,), dtype=object)
        i = 0
        # calculate for each reconstructed coordinates
        while i<n:
            binary = np.random.randint(2, size=(max_delay)).astype(bool)
            if np.count_nonzero(binary)!=0:
                delays = np.where(binary)[0]
                maximum = delays.max() + 1 # because zero is necessary
                result = np.zeros([n_timesteps - maximum, np.count_nonzero(binary) + 1])
                result[:,0] = x[maximum:] # our embedding system is t-tau type
                for j in range(result.shape[1]-1):
                    result[:,j+1] = x[maximum - delays[j] - 1:-delays[j] - 1]
                results[i] = result
                i += 1
    else:
        # embedding dimension is determined
        assert E <= max_delay
        results = np.zeros((n,), dtype=object)
        for i in range(n):
            combinations = np.sort(np.random.choice(max_delay, E-1, False))
            maximum = combinations.max() + 1
            result = np.zeros([n_timesteps - maximum, E])
            result[:,0] = x[maximum:]
            for j in range(result.shape[1]-1):
                result[:,j+1] = x[maximum - combinations[j]:-combinations[j]]
            results[i] = result
    if n==1:
        return results[0]
    else:
        return results



def generalized_delay_embedding(x, E=None, max_delay=5, n=1, seed=1):
    """
    Args:
        x [n_timesteps, n_dim]: data
        E {int}: embedding dimension
        max_delay {int}: maximum embedding dimension
        n {int}: number of embeddings for generating

    Return:
        [n, n_timesteps - delay, E] or [n_timesteps - delay, E]
    """
    np.random.seed(seed)
    n_timesteps, n_dim = x.shape

    # only maximum number of delay is determined
    if E is None:
        # binary array for embedding
        # if True, the correspoding delay interval is used.
        results = np.zeros((n,), dtype=object)
        count = 0
        # calculate for each reconstructed coordinates
        while count<n:
            binary = np.random.randint(2, size=(max_delay+1,n_dim)).astype(bool)
            delays, dims = np.where(binary)
            if delays.min()==0 and len(delays)>1:
                maximum = delays.max()
                result = np.zeros([n_timesteps - maximum, len(delays)])
                for j in range(len(delays)):
                    result[:,j] = x[maximum - delays[j]:n_timesteps - delays[j], dims[j]]
                results[count] = result
                count += 1
    else:
        # embedding dimension is determined
        assert E <= max_delay*n_dim
        results = np.zeros((n,), dtype=object)
        count = 0
        while count < n:
            combinations = np.sort(np.random.choice(max_delay*n_dim, E, False))
            dims, delays = combinations//max_delay, combinations%max_delay
            if delays.min()==0:
                maximum = delays.max()
                result = np.zeros([n_timesteps - maximum, E])
                for j in range(E):
                    result[:,j] = x[maximum - delays[j]:n_timesteps - delays[j], dims[j]]
                results[count] = result
                count += 1
    if n==1:
        return results[0]
    else:
        return results

