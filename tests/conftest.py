# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=too-many-locals, too-many-public-methods, no-member
# pylint: disable=too-many-arguments, too-many-branches, too-many-statements
"""Pytest configuration file defining helper functions."""

import os
import json
import pytest


def pytest_report_teststatus(report, config):  # pylint: disable=unused-argument
    """Specialize test reporting to allow atol violations to be reported"""
    if report.when == "call":
        if report.failed:
            return "failed", "F", "FAILED"
        if report.skipped:
            return "skipped", "S", "SKIPPED"
        if report.caplog != "" and "1E-05" not in report.caplog:
            setattr(report, "wasxfail", f"atol={report.caplog[-1]}")
            return "warning", report.caplog[-1], "WARNING"
        if report.passed:
            return "passed", ".", "PASSED"
        return "failed", "F", "FAILED"
    return None


if os.getenv("_PYTEST_TRACK_RESULTS", "0") != "0":
    RESULTS_FILE = "test_results.json"
    pytest.failed_nodes = {}

    @pytest.fixture(autouse=True, scope="module", name="track_results")
    def fixture_track_results():
        """Track previously run tests in RESULTS_FILE"""
        # Load existing results if the file exists
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "r", encoding="UTF-8") as filename:
                pytest.failed_nodes = json.load(filename)

        yield

        # Save results at the end of the session
        with open(RESULTS_FILE, "w", encoding="UTF-8") as filename:
            json.dump(pytest.failed_nodes, filename, indent=4)

    # Register a hook to capture test results after each test
    @pytest.hookimpl(tryfirst=True, hookwrapper=True)
    def pytest_runtest_makereport(item):
        """Check if user has requested termination, and store test status to list"""
        output = yield
        report = output.get_result()
        # Allow terminating early while recording the run so far tests
        if report.when == "teardown" and os.path.exists("stop"):
            os.remove("stop")
            raise RuntimeError("Terminating early")
        if report.when != "call":
            return
        if report.failed:
            pytest.failed_nodes[item.nodeid] = "failed"
        elif report.caplog != "" and "1E-05" not in report.caplog:
            pytest.failed_nodes[item.nodeid] = "atol_warning"
        else:
            pytest.failed_nodes[item.nodeid] = "passed"

    def should_skip_test(item):
        """Check if the test was previously passed"""
        return pytest.failed_nodes.get(item.nodeid) == "passed"

    @pytest.fixture(autouse=True)
    def skip_if_passed(request, track_results):  # pylint: disable=unused-argument
        """Skip a test if it was already ran and passed"""
        if should_skip_test(request.node):
            pytest.skip("Skipping previously passed test")


if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        """Raise interactive erros so debugger can capture them"""
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        """Raise pytest internal errors so debugger can capture them"""
        raise excinfo.value
