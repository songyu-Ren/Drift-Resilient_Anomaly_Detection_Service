from __future__ import annotations

import time
from collections.abc import Callable

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Dedicated registry to avoid leaking default metrics in tests
registry = CollectorRegistry()

requests_total = Counter(
    "requests_total",
    "Total number of requests by endpoint",
    labelnames=("endpoint",),
    registry=registry,
)

request_latency_seconds = Histogram(
    "request_latency_seconds",
    "Request latency by endpoint",
    labelnames=("endpoint",),
    registry=registry,
)

anomalies_total = Counter(
    "anomalies_total",
    "Total predicted anomalies by endpoint",
    labelnames=("endpoint",),
    registry=registry,
)

model_loaded = Gauge(
    "model_loaded",
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
