# SPDX-License-Identifier: LicenseRef-BrayanLopez-EvalOnly
"""Project identity must be stamped: removing it turns CI red."""
from pathlib import Path
from app.config import PROJECT_ID
PROJECT = "a084f047-317e-4529-9b51-56a900b6a152"
def test_project_identity_is_stamped() -> None:
    assert PROJECT_ID == PROJECT
def test_identity_is_in_the_sources() -> None:
    assert PROJECT in Path("data/sources.txt").read_text()
