import sys
from math import ceil
from collections import defaultdict

class KMeansEvenCluster(object):

    def __init__(self, size, label):
        self.label = label
        self.size = int(size)
        self.points = []
        self.centroid_sum = defaultdict(float) # {key: value_sum}
        self.centroid_keys = set() # {key}

    def __str__(self):
        return u'<Cluster %s>' % self.label

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.points == other.points

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.points < other.points

    def has_free_space(self):
        return len(self.points) < self.size

    def is_full(self):
        return len(self.points) >= self.size

    def distance2(self, point):
        """
        Calculates the euclidean distance-squared of the given point to the centroid.
        """
        if not self.points:
            print('distance2: no points in cluster %s' % self)
            return -1
        self.centroid_keys.update(point.keys())
        a = self.centroid_sum
        b = point
        d = sum((a[k]/float(len(self.points)) - b.get(k, 0))**2 for k in self.centroid_keys)
        return d

    def add(self, point):
        self.points.append(point)
        # Updated centroid.
        for k in point:
            self.centroid_sum[k] += point[k]

    def remove(self, point):
        assert point in self.points
        print('Removing point...')
        self.points.remove(point)
        # Update centroid.
        print('Updating centroid...')
        for k in point:
            self.centroid_sum[k] -= point[k]

    def pop(self):
        """
        Removes a point if the cluster is over-sized.
        """
        if len(self.points) <= self.size:
            return

        # Find and remove the point furtherest away from the centroid.
        best_score = None
        best_point = None
        print('Checking %i centroid keys...' % len(self.centroid_keys))
        for point in self.points:
            new_score = self.distance2(point)
            #print('best_score:', best_score)
            #print('point %s has score %s' % (id(point), new_score))
            #print('point:', point)
            if best_score is None or new_score > best_score:
                best_score = new_score
                best_point = point
        self.remove(best_point)

        # Refresh keys to purge unused values.
        print('Purging centroid keys...')
        self.centroid_keys = set()
        for point in self.points:
            self.centroid_keys.update(point.keys())
        print('New centroid keys:', len(self.centroid_keys))
        print('Done pop.')

        return best_point

class KMeansEven(object):

    def __init__(self, k):
        self.k = k
        self.clusters = []

    def find_best_cluster(self, point, last_cluster=None):
        """
        Finds the cluster with the smallest euclidean distance to its centroid.
        """
        best_score = None
        best_cluster = None
        for cluster in self.clusters:

            # If we just removed this point from a cluster, don't bother
            # checking to re-add it.
            if last_cluster and cluster is last_cluster:
                continue

            # Skip clusters that have no room.
            #TODO:fix? This was a workaround to prevent cases where a point
            # is equally fit for two clusters can infinitely keeps getting popped from one to the other.
            if cluster.is_full():
                continue

            new_score = cluster.distance2(point)
            #print('best score:', best_score)
            #print('Cluster %s has score %s.' % (cluster, new_score))
            if best_score is None or new_score < best_score:
                best_score = new_score
                best_cluster = cluster
        return best_cluster

    @property
    def clusters_with_free_space(self):
        i = 0
        for cluster in self.clusters:
            i += cluster.has_free_space()
        return i

    def fit(self, data, priors=None):
        # priors = [(point, index)]

        # Create initial empty clusters.
        size = int(ceil(len(data)/float(self.k)))
        for i in range(self.k):
            self.clusters.append(KMeansEvenCluster(size=size, label=i))
        print('total points: %i' % len(data))
        print('cluster: %i' % self.k)
        print('cluster size: %i' % size)
        assert self.k * size >= len(data)

        # Initialize cluster starting points.
        if priors:
            for point, index in priors:
                data.remove(point)
                self.clusters[index].add(point)

        # Put all data points into a cluster.
        data = list((_, None) for _ in data) # [(row, last_cluster)]
        recluster_count = 0
        cluster_key_size = 0
        iterations = 0
        while data:
            iterations += 1
            total = len(data)
            print('Cluster point distribution: %s' %  (', '.join(['%s: %i' % (cluster, len(cluster.points)) for cluster in self.clusters])))
            sys.stdout.write('\rIterations: %i. Points remaining to cluster: %i. Recluster count: %i. Cluster key size: %i. Clusters with free space: %i' \
                % (iterations, total, recluster_count, cluster_key_size, self.clusters_with_free_space))
            sys.stdout.flush()
            print('')
            point, last_cluster = data.pop(0)
            print('Finding best cluster for point %s which has %s features...' % (id(point), len(point)))
            cluster = self.find_best_cluster(point, last_cluster)
            print('Adding point to cluster %s...' % cluster)
            len0 = len(cluster.points)
            cluster.add(point)
            len1 = len(cluster.points)
            assert len1 > len0
            cluster_key_size = max(cluster_key_size, len(cluster.centroid_keys))
            cluster_key_size = max(cluster_key_size, len(point.keys()))
            print('Finding excess point...')
            new_point = cluster.pop()
            if new_point:
                recluster_count += 1
                data.append((new_point, cluster))
            print('')

        return self
