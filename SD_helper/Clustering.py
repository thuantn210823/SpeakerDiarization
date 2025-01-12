from typing import Any, Optional, Union

import torch
import numpy as np
import scipy
from sklearn.cluster import AgglomerativeClustering as AHC

from tqdm import tqdm
import collections

class ConstrainedAHC(AHC):
    def __init__(self,
                 linkage: str = 'average',
                 ml: Optional[list] = [],
                 cl: Optional[list] = [],
                 *args, **kwargs):
        if linkage == 'ward':
            metric = 'euclidean'
        else:
            metric = 'precomputed'
        super().__init__(metric = metric,
                         linkage = linkage,
                         *args, **kwargs)

        self.ml = ml
        self.cl = cl
    
    def fit(self,
            X: np.array,
            *args, **kwargs):
        if self.linkage == 'ward':
            return super()._fit(X)
        else:    
            X = scipy.spatial.distance.cdist(X, X, *args, **kwargs)
            for vs in self.ml:
                X[vs[0], vs[1]] = 0
                X[vs[1], vs[0]] = 0
            for vs in self.cl:
                X[vs[0], vs[1]] = 1e9
                X[vs[1], vs[0]] = 1e9
            return super()._fit(X)

class KMeans:
    def __init__(self,
                 n_clusters: int,
                 distance: Optional[Any] = None,
                 eps: float = 1e-5):
        self.n_clusters = n_clusters
        if distance is None or distance == 'euclid':
            self.cdist = self._distance
        else:
            self.cdist = distance
        self.eps = eps

    def __call__(self,
                 X: torch.Tensor,
                 num_iters: int):
        centroids = self._init_centroids(X)
        for i in tqdm(range(num_iters)):
            distance_mat = self.cdist(X, centroids)
            Y = self._update_labels(distance_mat)
            new_centroids = self._update_centroids(X, Y)
            if torch.norm(new_centroids - centroids, dim = -1).mean() < self.eps:
                break
            else:
                centroids = new_centroids
        return centroids, Y

    def _distance(self,
                  X: torch.Tensor,
                  centroids: torch.Tensor):
        """
        samples: (N, d)
        centroid: (m, d)
        """
        return (torch.sqrt(((X.unsqueeze(0) - centroids.unsqueeze(1))**2).sum(dim = -1))).transpose(0, 1)

    def _init_centroids(self,
                        X: torch.Tensor):
        centroids = X[torch.randint(0, X.shape[0], (self.n_clusters,))]
        return centroids

    def _update_labels(self,
                       distance_mat: torch.Tensor):
        return torch.argmin(distance_mat, dim = -1)

    def _update_centroids(self,
                          X: torch.Tensor,
                          Y: torch.Tensor):
        centroids = []
        for i in range(self.n_clusters):
            centroids.append(torch.mean(X[Y == i], dim = 0))
        return torch.stack(centroids)

class COPKMeans(KMeans):
    def __init__(self,
                 n_clusters: int,
                 distance: Optional[Any] = None,
                 ml: Optional[list] = [],
                 cl: Optional[list] = [],
                 *args, **kwargs):
        """
        Parameters:
        - n_cluster: int
          Number of clusters
        - distance: Optional[Any], default: None
        - ml: Optional[list], default: empty list
          Must-link constrain
        - cl: Optional[list], default: empty list
          Cannot-link constrain 
        """
        super().__init__(n_clusters,
                         distance,
                         *args, **kwargs)
        self.ml = ml
        self.cl = cl

    def __call__(self,
                 X: torch.Tensor,
                 num_iters: int):
        if len(self.ml) + len(self.cl) == 0:
            return super().__call__(X, num_iters)
        else:
            return self._constrained_kmeans(X, num_iters)
        
    def _constrained_kmeans(self,
                           X: torch.Tensor, 
                           num_iters: int):
        self._make_adj_list()
        centroids = self._init_centroids(X)
        Y = torch.tensor([-1]*X.shape[0])
        for i in range(num_iters):
            for idx, x in enumerate(X):
                _, cluster_indices = torch.sort(torch.cdist(x.unsqueeze(0), centroids).squeeze(0))
                is_assigned = False
                for cluster_idx in cluster_indices:
                    if not self._violate_constraints(idx, cluster_idx, Y):
                        Y[idx] = cluster_idx
                        is_assigned = True
                        break
                if not is_assigned:
                    return None, None
            new_centroids = self._update_centroids(X, Y)
            if torch.norm(new_centroids - centroids, dim = -1).mean() < self.eps:
                return new_centroids, Y
            else:
                centroids = new_centroids
        return centroids, Y
            
    def _violate_constraints(self,
                             data_idx: Union[int, torch.Tensor],
                             cluster_idx: Union[int, torch.Tensor],
                             labels: torch.Tensor):
        for v in self.ml[data_idx]:
              if labels[v] != cluster_idx  and labels[v] != -1:
                    return True
        for v in self.cl[data_idx]:
              if labels[v] == cluster_idx and labels[v] != -1:
                    return True
        return False
    
    def _make_adj_list(self):
        ml = collections.defaultdict(set)
        cl = collections.defaultdict(set)
        for vs in self.ml:
            ml[vs[0]].add(vs[1])
            ml[vs[1]].add(vs[0])
        for vs in self.cl:
            cl[vs[0]].add(vs[1])
            cl[vs[1]].add(vs[0])
        self.ml = ml
        self.cl = cl
