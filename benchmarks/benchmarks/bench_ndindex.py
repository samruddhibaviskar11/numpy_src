"""
ASV benchmark for ndindex performance.

This benchmark compares the performance and memory usage of `numpy.ndindex`
against a direct `itertools.product` equivalent.

To run locally:
1. Navigate to the NumPy root directory.
2. Run: `asv run -b NdIndexBenchmark`
"""
import numpy as np
from itertools import product
from numpy.lib import ndindex


class NdIndexBenchmark:
    # Benchmark will be run for each shape in this list
    params = [
        (10, 10),
        (20, 20),
        (50, 50),
        (10, 10, 10),
        (20, 30, 40),
        (50, 60, 90)
    ]
    param_names = ['shape']

    def time_ndindex(self, shape):
        """Time the performance of np.ndindex (iterator consumption)."""
        for _ in ndindex(*shape):
            pass

    def time_itertools_product(self, shape):
        """Time the performance of itertools.product (baseline)."""
        for _ in product(*(range(s) for s in shape)):
            pass

    def peakmem_ndindex(self, shape):
        """Measure peak memory for creating and consuming np.ndindex."""
        return list(ndindex(*shape))

    def peakmem_itertools_product(self, shape):
        """Measure peak memory for creating and consuming itertools.product."""
        return list(product(*(range(s) for s in shape)))
