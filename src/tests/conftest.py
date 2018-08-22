# The following code taken from
# https://docs.pytest.org/en/latest/example/simple.html
    #control-skipping-of-tests-according-to-command-line-option
# on 2018-08-22. Assumed that it's for re=use as it's in the `Examples'
# section of the website, or it's under the MIT licence. Need to clarify
# this.

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
