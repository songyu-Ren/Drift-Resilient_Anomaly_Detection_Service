from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from .settings import get_settings

# Dedicated registry to avoid leaking default metrics in tests
registry = CollectorRegistry()

# Load latency buckets from settings for sensible histogram ranges
_latency_buckets = tuple(get_settings().monitoring.latency_buckets)

requests_total = Counter(
    "dds_requests_total",
    "Total number of requests",
    labelnames=("endpoint", "method", "status"),
    registry=registry,
)

request_latency_seconds = Histogram(
    "dds_request_latency_seconds",
    "Request latency by endpoint",
    labelnames=("endpoint",),
    buckets=_latency_buckets,
    registry=registry,
)

anomalies_total = Counter(
    "dds_anomalies_total",
    "Total predicted anomalies by endpoint",
    labelnames=("endpoint",),
    registry=registry,
)

model_loaded = Gauge(
    "dds_model_loaded",
    "Gauge set to 1 when model is loaded, 0 otherwise",
    registry=registry,
)


def metrics_exposition_text() -> bytes:
    return generate_latest(registry)


def measure_latency(endpoint: str) -> Callable:
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                request_latency_seconds.labels(endpoint=endpoint).observe(duration)

        return wrapper

    return decorator


@contextmanager
def latency_timer(endpoint: str):
    """Context manager to time endpoint latency and observe histogram."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        request_latency_seconds.labels(endpoint=endpoint).observe(duration)
