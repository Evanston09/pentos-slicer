from contextlib import ExitStack

import pytest

from services.slice_jobs import SlicingBusyError, SlicingCoordinator


def test_coordinator_allows_two_jobs_and_rejects_third() -> None:
    coordinator = SlicingCoordinator(2)

    with ExitStack() as stack:
        stack.enter_context(coordinator.slot())
        stack.enter_context(coordinator.slot())

        with pytest.raises(SlicingBusyError, match="Server is busy"):
            with coordinator.slot():
                pass

    with coordinator.slot():
        pass
