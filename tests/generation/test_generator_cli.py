from __future__ import annotations

import json
from pathlib import Path

import yaml

from efd.generation.__main__ import main
from efd.network import METADATA_FILE, TABLE_FILES, InvoiceNetwork

from generation_helpers import small_raw


def test_command_writes_scenario(tmp_path: Path) -> None:
    config_path = tmp_path / "generator.yaml"
    config_path.write_text(yaml.safe_dump(small_raw(), allow_unicode=True), encoding="utf-8")

    directory = main(["--config", str(config_path), "--output-root", str(tmp_path / "out")])

    assert directory == tmp_path / "out" / "prev1_medio_s42"
    assert all((directory / file).exists() for file in TABLE_FILES.values())
    metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
    network = InvoiceNetwork.load(directory)
    assert metadata["seed"] == 42
    assert metadata["counts"]["taxpayers"] == len(network.taxpayers) == 2000
    assert metadata["counts"]["invoices"] == len(network.invoices)
    assert metadata["counts"]["shells_by_pattern"] == {
        "isolated_emitter": 8,
        "ring": 4,
        "chain": 4,
        "shared_representative": 4,
    }
    assert set(metadata) >= {"config", "config_hash", "library_versions"}
