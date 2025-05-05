"""Script to run the benchmarks."""

# pylint: disable=invalid-name

import sys
from benchmarks.benchmark import Benchmark
from benchmarks.efficient_su2 import run_efficientsu2
""" from benchmarks.featuremap import run_featuremap
from benchmarks.maxcut import run_maxcut """


run_efficientsu2()