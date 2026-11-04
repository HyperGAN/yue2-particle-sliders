"""Keep immutable historical CDF checks in their declared dependency cohort."""
from importlib import metadata

import pytest


def pytest_collection_modifyitems(items):
    try:
        legacy = (metadata.version('particle-sliders-core'), metadata.version('particlegan')) == ('0.6.0', '0.9.1')
    except metadata.PackageNotFoundError:
        legacy = False
    if not legacy:
        marker = pytest.mark.skip(reason='historical CDF check requires its original core0.6/PG0.9.1 environment')
        for item in items:
            if item.get_closest_marker('legacy_pg091'):
                item.add_marker(marker)
