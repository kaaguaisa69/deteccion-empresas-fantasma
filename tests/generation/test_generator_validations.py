"""Las validaciones de docs/generator_design.md sobre un escenario de 2 000 contribuyentes.

Cada validación se comprueba en el escenario generado (en memoria y, cuando aplica, leído de disco) y
con una alteración que debe hacerla fallar.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from efd.generation import GeneratedScenario, GeneratorConfig, InvoiceNetworkGenerator
from efd.generation.validation import (
    ValidationError,
    check_heavy_tail,
    check_invoice_dates,
    check_invoice_integrity,
    check_legitimate_noise,
    check_pattern_structures,
    check_recent_legitimate,
    check_round_legitimate_amounts,
    check_shell_counts,
    validate_network,
)
from efd.network import METADATA_FILE, TABLE_FILES, InvoiceNetwork

from generation_helpers import small_raw


def _ids(network: InvoiceNetwork, pattern: str) -> set[str]:
    taxpayers = network.taxpayers
    return set(taxpayers.loc[taxpayers["pattern"] == pattern, "taxpayer_id"])


def _drop_invoices(network: InvoiceNetwork, mask: pd.Series) -> InvoiceNetwork:
    return replace(network, invoices=network.invoices[~mask].reset_index(drop=True))


def test_full_validation_passes(small_scenario: GeneratedScenario, small_config: GeneratorConfig) -> None:
    validate_network(small_scenario.network, small_config, small_scenario.active_until)


def test_written_files_pass_validation(written_dir: Path, small_config: GeneratorConfig) -> None:
    validate_network(InvoiceNetwork.load(written_dir), small_config)


# 1. Misma semilla y configuración producen archivos idénticos.


def test_same_seed_produces_identical_files(
    written_dir: Path, small_config: GeneratorConfig, tmp_path: Path
) -> None:
    second = InvoiceNetworkGenerator(small_config).run(tmp_path)
    for file in [*TABLE_FILES.values(), METADATA_FILE]:
        assert (written_dir / file).read_bytes() == (second / file).read_bytes(), file


def test_different_seed_produces_different_network(written_dir: Path, tmp_path: Path) -> None:
    raw = small_raw()
    raw["seed"] = 7
    other = InvoiceNetworkGenerator(GeneratorConfig.from_dict(raw)).run(tmp_path)
    invoices = TABLE_FILES["invoices"]
    assert (written_dir / invoices).read_bytes() != (other / invoices).read_bytes()


# 2. Número exacto de fantasmas y proporción por patrón.


def test_shell_counts(small_scenario: GeneratedScenario, small_config: GeneratorConfig) -> None:
    taxpayers = small_scenario.network.taxpayers
    assert int(taxpayers["is_shell"].sum()) == 20
    assert taxpayers.loc[taxpayers["is_shell"] == 1, "pattern"].value_counts().to_dict() == {
        "isolated_emitter": 8,
        "ring": 4,
        "chain": 4,
        "shared_representative": 4,
    }
    check_shell_counts(small_scenario.network, small_config)

    relabeled = taxpayers.copy()
    relabeled.loc[relabeled["pattern"] == "ring", "pattern"] = "chain"
    with pytest.raises(ValidationError):
        check_shell_counts(replace(small_scenario.network, taxpayers=relabeled), small_config)


# 3. Anillos en ciclo, cadenas de 2 a 4 saltos y grupos P4 con representante compartido.


def test_rings_form_cycles(small_scenario: GeneratedScenario, small_config: GeneratorConfig) -> None:
    network = small_scenario.network
    ring = _ids(network, "ring")
    invoices = network.invoices
    internal = invoices[invoices["issuer_id"].isin(ring) & invoices["receiver_id"].isin(ring)]
    graph = nx.DiGraph(internal[["issuer_id", "receiver_id"]].itertuples(index=False))
    assert set(graph) == ring
    assert all(3 <= len(cycle) <= 6 for cycle in nx.simple_cycles(graph))
    check_pattern_structures(network, small_config)

    edge = internal.iloc[0]
    broken = (invoices["issuer_id"] == edge["issuer_id"]) & (invoices["receiver_id"] == edge["receiver_id"])
    with pytest.raises(ValidationError, match="anillo"):
        check_pattern_structures(_drop_invoices(network, broken), small_config)


def test_chains_have_two_to_four_hops(
    small_scenario: GeneratedScenario, small_config: GeneratorConfig
) -> None:
    network = small_scenario.network
    chain = _ids(network, "chain")
    invoices = network.invoices
    internal = invoices[invoices["issuer_id"].isin(chain) & invoices["receiver_id"].isin(chain)]
    graph = nx.DiGraph(internal[["issuer_id", "receiver_id"]].itertuples(index=False))
    graph.add_nodes_from(chain)
    for component in nx.weakly_connected_components(graph):
        assert 2 <= len(component) <= 4
        assert nx.is_directed_acyclic_graph(graph.subgraph(component))

    edge = internal.iloc[0]
    broken = (invoices["issuer_id"] == edge["issuer_id"]) & (invoices["receiver_id"] == edge["receiver_id"])
    with pytest.raises(ValidationError, match="cadena"):
        check_pattern_structures(_drop_invoices(network, broken), small_config)


def test_shared_representative_groups(
    small_scenario: GeneratedScenario, small_config: GeneratorConfig
) -> None:
    network = small_scenario.network
    p4 = _ids(network, "shared_representative")
    links = network.representatives
    own = links[links["taxpayer_id"].isin(p4)]
    assert set(own["taxpayer_id"]) == p4
    assert own["representative_id"].nunique() == 1  # cuatro fantasmas P4 forman un grupo
    taxpayers = network.taxpayers.set_index("taxpayer_id")
    assert (taxpayers.loc[sorted(p4), "taxpayer_type"] == "sociedad").all()

    split = links.copy()
    split.loc[split["taxpayer_id"].isin(p4), "representative_id"] = [
        f"RL-X{i}" for i in range(len(p4))
    ]
    with pytest.raises(ValidationError, match="representante"):
        check_pattern_structures(replace(network, representatives=split), small_config)


# 4. Sin autofacturas; IVA y total coherentes.


def test_invoice_integrity(small_scenario: GeneratedScenario, small_config: GeneratorConfig) -> None:
    network = small_scenario.network
    invoices = network.invoices
    assert (invoices["issuer_id"] != invoices["receiver_id"]).all()
    np.testing.assert_array_equal(invoices["vat"], np.round(0.15 * invoices["subtotal_15"], 2))
    np.testing.assert_allclose(
        invoices["total"], invoices["subtotal_0"] + invoices["subtotal_15"] + invoices["vat"], atol=1e-6
    )
    assert (invoices["subtotal_0"] > 0).any()  # existen bienes con tarifa 0 %
    check_invoice_integrity(network, small_config)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda inv: inv.assign(vat=inv["vat"] + 0.01),
        lambda inv: inv.assign(total=inv["total"] + 1.0),
        lambda inv: inv.assign(receiver_id=inv["issuer_id"]),
    ],
)
def test_invoice_integrity_detects_tampering(
    small_scenario: GeneratedScenario, small_config: GeneratorConfig, tamper
) -> None:
    network = small_scenario.network
    with pytest.raises(ValidationError):
        check_invoice_integrity(replace(network, invoices=tamper(network.invoices)), small_config)


# 5. Fechas dentro del período, tras el inicio del emisor y dentro de la vida de cada P1.


def test_invoice_dates(small_scenario: GeneratedScenario, small_config: GeneratorConfig) -> None:
    network = small_scenario.network
    invoices = network.invoices
    start = network.taxpayers.set_index("taxpayer_id")["start_date"]
    assert invoices["issue_date"].between(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")).all()
    assert (invoices["issue_date"] >= start.loc[invoices["issuer_id"]].to_numpy()).all()

    p1 = _ids(network, "isolated_emitter")
    life = small_scenario.active_until.loc[sorted(p1)] - start.loc[sorted(p1)]
    assert life.between(pd.Timedelta(days=88), pd.Timedelta(days=275)).all()  # de 3 a 9 meses
    for column in ("issuer_id", "receiver_id"):
        involved = invoices[invoices[column].isin(p1)]
        assert (involved["issue_date"] <= small_scenario.active_until.loc[involved[column]].to_numpy()).all()

    check_invoice_dates(network, small_config, small_scenario.active_until)
    check_invoice_dates(network, small_config)


def test_invoice_dates_detect_activity_outside_life(
    small_scenario: GeneratedScenario, small_config: GeneratorConfig
) -> None:
    network = small_scenario.network
    active_until = small_scenario.active_until
    p1 = _ids(network, "isolated_emitter")
    ends_early = [shell for shell in sorted(p1) if active_until.loc[shell] < pd.Timestamp("2025-12-31")]
    assert ends_early, "Ningún P1 termina antes del fin del período; la prueba no tendría efecto."
    invoices = network.invoices.copy()
    row = invoices.index[invoices["issuer_id"] == ends_early[0]][0]
    invoices.loc[row, "issue_date"] = active_until.loc[ends_early[0]] + pd.Timedelta(days=1)
    with pytest.raises(ValidationError, match="P1"):
        check_invoice_dates(replace(network, invoices=invoices), small_config, small_scenario.active_until)

    early = network.invoices.copy()
    early.loc[row, "issue_date"] = pd.Timestamp("2024-12-31")
    with pytest.raises(ValidationError, match="período"):
        check_invoice_dates(replace(network, invoices=early), small_config)


# 6. Distribución de grado de cola pesada.


def test_heavy_tailed_degree(small_scenario: GeneratedScenario, small_config: GeneratorConfig) -> None:
    check_heavy_tail(small_scenario.network, small_config)


def test_heavy_tail_rejects_regular_network(small_config: GeneratorConfig) -> None:
    ids = [f"TP{i}" for i in range(30)]
    taxpayers = pd.DataFrame(
        {
            "taxpayer_id": ids,
            "taxpayer_type": "sociedad",
            "sector": "comercio",
            "province": "Guayas",
            "start_date": pd.Timestamp("2020-01-01"),
            "size": "micro",
            "is_shell": 0,
            "pattern": None,
        }
    )
    invoices = pd.DataFrame(
        {
            "invoice_id": [f"F{i}" for i in range(30)],
            "issuer_id": ids,
            "receiver_id": ids[1:] + ids[:1],
            "issue_date": pd.Timestamp("2025-06-01"),
            "subtotal_0": 0.0,
            "subtotal_15": 100.0,
            "vat": 15.0,
            "total": 115.0,
        }
    )
    representatives = pd.DataFrame({"representative_id": ["RL1"], "taxpayer_id": ["TP0"]})
    with pytest.raises(ValidationError, match="Grado"):
        check_heavy_tail(InvoiceNetwork(taxpayers, invoices, representatives), small_config)


# 7. Grupos legítimos con representante compartido, comercio recíproco y ciclos legítimos.


def test_legitimate_noise(small_scenario: GeneratedScenario, small_config: GeneratorConfig) -> None:
    network = small_scenario.network
    check_legitimate_noise(network, small_config)

    shells = set(network.taxpayers.loc[network.taxpayers["is_shell"] == 1, "taxpayer_id"])
    links = network.representatives
    legit_links = links[~links["taxpayer_id"].isin(shells)]
    assert (legit_links["representative_id"].value_counts() > 1).any()

    without_groups = links.assign(representative_id=[f"RL-U{i}" for i in range(len(links))])
    with pytest.raises(ValidationError, match="representante"):
        check_legitimate_noise(replace(network, representatives=without_groups), small_config)


def test_legitimate_noise_requires_reciprocal_trade(
    small_scenario: GeneratedScenario, small_config: GeneratorConfig
) -> None:
    network = small_scenario.network
    invoices = network.invoices
    pairs = invoices[["issuer_id", "receiver_id"]]
    # Conserva un solo sentido de cada par: elimina las facturas del emisor "mayor".
    reciprocal = pairs.merge(
        pairs.rename(columns={"issuer_id": "receiver_id", "receiver_id": "issuer_id"}).drop_duplicates(),
        how="left",
        indicator=True,
    )["_merge"].eq("both").to_numpy()
    mask = pd.Series(reciprocal & (invoices["issuer_id"] > invoices["receiver_id"]).to_numpy())
    with pytest.raises(ValidationError, match="recíproco"):
        check_legitimate_noise(_drop_invoices(network, mask), small_config)


# 8. Legítimos con inicio de actividades dentro del período.


def test_recent_legitimate_taxpayers(
    small_scenario: GeneratedScenario, small_config: GeneratorConfig
) -> None:
    network = small_scenario.network
    taxpayers = network.taxpayers
    recent = taxpayers["start_date"] >= pd.Timestamp("2025-01-01")
    assert (recent & (taxpayers["is_shell"] == 0)).sum() > 0
    check_recent_legitimate(network, small_config)

    old = taxpayers.copy()
    old.loc[old["is_shell"] == 0, "start_date"] = pd.Timestamp("2010-01-01")
    with pytest.raises(ValidationError, match="recientes"):
        check_recent_legitimate(replace(network, taxpayers=old), small_config)


# 9. Facturas legítimas con montos redondos.


def test_round_legitimate_amounts(
    small_scenario: GeneratedScenario, small_config: GeneratorConfig
) -> None:
    network = small_scenario.network
    check_round_legitimate_amounts(network, small_config)

    shells = set(network.taxpayers.loc[network.taxpayers["is_shell"] == 1, "taxpayer_id"])
    invoices = network.invoices.copy()
    legit = ~invoices["issuer_id"].isin(shells) & ~invoices["receiver_id"].isin(shells)
    invoices.loc[legit, "subtotal_15"] += 0.37
    with pytest.raises(ValidationError, match="redondas"):
        check_round_legitimate_amounts(replace(network, invoices=invoices), small_config)


def test_validate_network_reports_every_failure(
    small_scenario: GeneratedScenario, small_config: GeneratorConfig
) -> None:
    network = small_scenario.network
    invoices = network.invoices.assign(vat=network.invoices["vat"] + 0.01)
    taxpayers = network.taxpayers.copy()
    taxpayers.loc[taxpayers["is_shell"] == 0, "start_date"] = pd.Timestamp("2010-01-01")
    with pytest.raises(ValidationError) as error:
        validate_network(replace(network, invoices=invoices, taxpayers=taxpayers), small_config)
    assert "vat" in str(error.value) and "recientes" in str(error.value)
