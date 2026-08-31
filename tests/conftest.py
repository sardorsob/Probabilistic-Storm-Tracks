import json
from pathlib import Path

import pytest


@pytest.fixture
def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def legacy_contract(repository_root: Path) -> dict[str, dict[str, object]]:
    contract_path = repository_root / "tests" / "fixtures" / "legacy_dashboard_contract.json"
    with contract_path.open(encoding="utf-8") as contract_file:
        return json.load(contract_file)
