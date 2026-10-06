"""Comprobaciones de realismo e integridad de una red generada.

Cada función implementa una validación de ``docs/generator_design.md`` y lanza ``ValidationError`` si
no se cumple. La reproducibilidad (misma semilla, mismos archivos) se verifica en las pruebas, porque
requiere dos ejecuciones.
"""

from __future__ import annotations

from collections.abc import Callable

import networkx as nx
import numpy as np
import pandas as pd

from efd.generation.config import GeneratorConfig
from efd.generation.patterns.chain import ChainPattern
from efd.generation.patterns.isolated_emitter import IsolatedEmitterPattern
from efd.generation.patterns.ring import RingPattern
from efd.generation.patterns.shared_representative import SharedRepresentativePattern
from efd.network import InvoiceNetwork

HEAVY_TAIL_RATIO = 10


class ValidationError(ValueError):
    """La red generada no cumple una validación de realismo o integridad."""


def _shell_ids(network: InvoiceNetwork) -> pd.Index:
    taxpayers = network.taxpayers
    return pd.Index(taxpayers.loc[taxpayers["is_shell"] == 1, "taxpayer_id"])


def _pattern_ids(network: InvoiceNetwork, pattern: str) -> pd.Index:
    taxpayers = network.taxpayers
    return pd.Index(taxpayers.loc[taxpayers["pattern"] == pattern, "taxpayer_id"])


def _legit_invoices(network: InvoiceNetwork) -> pd.DataFrame:
    shells = _shell_ids(network)
    invoices = network.invoices
    return invoices[~invoices["issuer_id"].isin(shells) & ~invoices["receiver_id"].isin(shells)]


def _pattern_graph(network: InvoiceNetwork, pattern: str) -> nx.DiGraph:
    """Grafo dirigido de las facturas entre fantasmas de un mismo patrón."""
    ids = _pattern_ids(network, pattern)
    invoices = network.invoices
    internal = invoices[invoices["issuer_id"].isin(ids) & invoices["receiver_id"].isin(ids)]
    graph = nx.DiGraph()
    graph.add_nodes_from(ids)
    graph.add_edges_from(internal[["issuer_id", "receiver_id"]].drop_duplicates().itertuples(index=False))
    return graph


def check_shell_counts(network: InvoiceNetwork, config: GeneratorConfig) -> None:
    """Validación 2: número exacto de fantasmas y reparto exacto por patrón."""
    taxpayers = network.taxpayers
    shells = taxpayers[taxpayers["is_shell"] == 1]
    if len(shells) != config.n_shells:
        raise ValidationError(f"Hay {len(shells)} fantasmas; se configuraron {config.n_shells}.")
    if taxpayers.loc[taxpayers["is_shell"] == 0, "pattern"].notna().any():
        raise ValidationError("Hay contribuyentes legítimos con patrón asignado.")
    observed = shells["pattern"].value_counts().to_dict()
    expected = {name: count for name, count in config.shell_counts().items() if count > 0}
    if observed != expected:
        raise ValidationError(f"Fantasmas por patrón {observed}; se esperaban {expected}.")


def check_pattern_structures(network: InvoiceNetwork, config: GeneratorConfig) -> None:
    """Validación 3: anillos P2 en ciclo, cadenas P3 de 2 a 4 saltos y grupos P4 con representante."""
    if RingPattern.name in config.patterns:
        low, high = config.patterns[RingPattern.name].params["group_size"]
        graph = _pattern_graph(network, RingPattern.name)
        for component in nx.weakly_connected_components(graph):
            ring = graph.subgraph(component)
            is_cycle = (
                all(ring.in_degree(node) == 1 and ring.out_degree(node) == 1 for node in ring)
                and nx.is_strongly_connected(ring)
            )
            if not (is_cycle and low <= len(ring) <= high):
                raise ValidationError(f"El anillo {sorted(component)} no forma un ciclo de {low} a {high}.")

    if ChainPattern.name in config.patterns:
        low, high = config.patterns[ChainPattern.name].params["hops"]
        graph = _pattern_graph(network, ChainPattern.name)
        shells = _shell_ids(network)
        invoices = network.invoices
        for component in nx.weakly_connected_components(graph):
            chain = graph.subgraph(component)
            is_path = (
                nx.is_directed_acyclic_graph(chain)
                and chain.number_of_edges() == len(chain) - 1
                and all(chain.in_degree(node) <= 1 and chain.out_degree(node) <= 1 for node in chain)
            )
            last = [node for node in chain if chain.out_degree(node) == 0]
            reaches_beneficiary = bool(last) and (
                ~invoices.loc[invoices["issuer_id"] == last[0], "receiver_id"].isin(shells)
            ).any()
            # Una cadena de h fantasmas tiene h - 1 saltos internos más el salto al beneficiario.
            if not (is_path and reaches_beneficiary and low <= len(chain) <= high):
                raise ValidationError(f"La cadena {sorted(component)} no tiene de {low} a {high} saltos.")

    if SharedRepresentativePattern.name in config.patterns:
        low, high = config.patterns[SharedRepresentativePattern.name].params["group_size"]
        ids = _pattern_ids(network, SharedRepresentativePattern.name)
        links = network.representatives
        own = links[links["taxpayer_id"].isin(ids)]
        if len(own) != len(ids) or own["taxpayer_id"].duplicated().any():
            raise ValidationError("Cada fantasma P4 debe tener exactamente un representante.")
        groups = links[links["representative_id"].isin(own["representative_id"])]
        for representative, members in groups.groupby("representative_id")["taxpayer_id"]:
            if not members.isin(ids).all() or not low <= len(members) <= high:
                raise ValidationError(
                    f"El representante {representative} no agrupa de {low} a {high} fantasmas P4."
                )


def check_invoice_integrity(network: InvoiceNetwork, config: GeneratorConfig) -> None:
    """Validación 4: sin autofacturas; IVA y total coherentes con los subtotales."""
    invoices = network.invoices
    if (invoices["issuer_id"] == invoices["receiver_id"]).any():
        raise ValidationError("Existen autofacturas.")
    expected_vat = np.round(config.market.vat_rate * invoices["subtotal_15"].to_numpy(), 2)
    if not np.allclose(invoices["vat"].to_numpy(), expected_vat, rtol=0, atol=1e-9):
        raise ValidationError("Hay facturas con vat distinto de round(tarifa × subtotal_15, 2).")
    expected_total = invoices[["subtotal_0", "subtotal_15", "vat"]].sum(axis=1).to_numpy()
    if not np.allclose(invoices["total"].to_numpy(), expected_total, rtol=0, atol=1e-6):
        raise ValidationError("Hay facturas con total distinto de subtotal_0 + subtotal_15 + vat.")
    if (invoices[["subtotal_0", "subtotal_15"]].to_numpy() < 0).any():
        raise ValidationError("Hay subtotales negativos.")


def check_invoice_dates(
    network: InvoiceNetwork, config: GeneratorConfig, active_until: pd.Series | None = None
) -> None:
    """Validación 5: fechas dentro del período y de la actividad de emisor, receptor y P1.

    Con ``active_until`` (último día de actividad por contribuyente, disponible al generar) se exige
    que un P1 no facture fuera de su vida. Sobre archivos leídos de disco, que no incluyen esa fecha,
    se exige que no facture después de ``start_date`` más la vida máxima configurada.
    """
    invoices = network.invoices
    start = network.taxpayers.set_index("taxpayer_id")["start_date"]
    dates = invoices["issue_date"]
    if dates.min() < pd.Timestamp(config.period.start) or dates.max() > pd.Timestamp(config.period.end):
        raise ValidationError("Hay facturas fuera del período simulado.")
    for column in ("issuer_id", "receiver_id"):
        if (dates < start.loc[invoices[column]].to_numpy()).any():
            raise ValidationError(f"Hay facturas anteriores al inicio de actividades de '{column}'.")

    if IsolatedEmitterPattern.name not in config.patterns:
        return
    p1 = _pattern_ids(network, IsolatedEmitterPattern.name)
    if active_until is None:
        max_life = config.patterns[IsolatedEmitterPattern.name].params["life_months"][1]
        limit = start.loc[p1] + pd.DateOffset(months=max_life) - pd.Timedelta(days=1)
    else:
        limit = active_until.loc[p1]
    for column in ("issuer_id", "receiver_id"):
        involved = invoices[invoices[column].isin(p1)]
        if (involved["issue_date"] > limit.loc[involved[column]].to_numpy()).any():
            raise ValidationError("Hay emisores aislados (P1) que facturan fuera de su vida.")


def check_heavy_tail(network: InvoiceNetwork, config: GeneratorConfig) -> None:
    """Validación 6: grado máximo mayor que ``HEAVY_TAIL_RATIO`` veces la mediana.

    El grado es el número de contrapartes distintas de cada contribuyente, sin considerar la
    dirección de las facturas.
    """
    ids = pd.Index(network.taxpayers["taxpayer_id"])
    issuer = ids.get_indexer(network.invoices["issuer_id"])
    receiver = ids.get_indexer(network.invoices["receiver_id"])
    pairs = np.unique(np.minimum(issuer, receiver) * len(ids) + np.maximum(issuer, receiver))
    degree = np.bincount(pairs // len(ids), minlength=len(ids)) + np.bincount(
        pairs % len(ids), minlength=len(ids)
    )
    if not degree.max() > HEAVY_TAIL_RATIO * np.median(degree):
        raise ValidationError(
            f"Grado máximo {degree.max()} no supera {HEAVY_TAIL_RATIO} veces la mediana {np.median(degree)}."
        )


def check_legitimate_noise(network: InvoiceNetwork, config: GeneratorConfig) -> None:
    """Validación 7: grupos legítimos con representante compartido, comercio recíproco y ciclos."""
    shells = _shell_ids(network)
    links = network.representatives
    legit_links = links[~links["taxpayer_id"].isin(shells)]
    if not (legit_links["representative_id"].value_counts() > 1).any():
        raise ValidationError("No hay grupos legítimos con representante compartido.")

    pairs = _legit_invoices(network)[["issuer_id", "receiver_id"]].drop_duplicates()
    reverse = pairs.rename(columns={"issuer_id": "receiver_id", "receiver_id": "issuer_id"})
    reciprocal = pairs.merge(reverse, on=["issuer_id", "receiver_id"], how="left", indicator=True)
    is_reciprocal = (reciprocal["_merge"] == "both").to_numpy()
    if not is_reciprocal.any():
        raise ValidationError("No hay comercio recíproco entre contribuyentes legítimos.")

    graph = nx.DiGraph(pairs[~is_reciprocal].itertuples(index=False))
    if nx.is_directed_acyclic_graph(graph):
        raise ValidationError("No hay ciclos legítimos de tres o más contribuyentes.")


def check_recent_legitimate(network: InvoiceNetwork, config: GeneratorConfig) -> None:
    """Validación 8: legítimos que inician en el período, al menos tantos como fantasmas recientes.

    Así una regla basada solo en la antigüedad no separa a las fantasmas.
    """
    taxpayers = network.taxpayers
    recent = taxpayers["start_date"] >= pd.Timestamp(config.period.start)
    recent_legit = int((recent & (taxpayers["is_shell"] == 0)).sum())
    recent_shells = int((recent & (taxpayers["is_shell"] == 1)).sum())
    if recent_legit == 0 or recent_legit < recent_shells:
        raise ValidationError(
            f"Legítimos recientes ({recent_legit}) menos que fantasmas recientes ({recent_shells})."
        )


def _is_round(invoices: pd.DataFrame, multiple: int) -> np.ndarray:
    cents = np.rint((invoices["subtotal_0"] + invoices["subtotal_15"]).to_numpy() * 100).astype(np.int64)
    return cents % (multiple * 100) == 0


def check_round_legitimate_amounts(network: InvoiceNetwork, config: GeneratorConfig) -> None:
    """Validación 9: facturas legítimas con montos redondos, al menos tantas como las de fantasmas.

    Así una regla basada solo en montos redondos no separa a las fantasmas.
    """
    multiple = min(config.market.round_multiples)
    legit_round = int(_is_round(_legit_invoices(network), multiple).sum())
    invoices = network.invoices
    shell_issued = invoices[invoices["issuer_id"].isin(_shell_ids(network))]
    shell_round = int(_is_round(shell_issued, multiple).sum())
    if legit_round == 0 or legit_round < shell_round:
        raise ValidationError(
            f"Facturas legítimas redondas ({legit_round}) menos que las de fantasmas ({shell_round})."
        )


CHECKS: tuple[Callable[[InvoiceNetwork, GeneratorConfig], None], ...] = (
    check_shell_counts,
    check_pattern_structures,
    check_invoice_integrity,
    check_heavy_tail,
    check_legitimate_noise,
    check_recent_legitimate,
    check_round_legitimate_amounts,
)


def validate_network(
    network: InvoiceNetwork, config: GeneratorConfig, active_until: pd.Series | None = None
) -> None:
    """Ejecuta todas las validaciones y reporta juntas las que fallen.

    Raises:
        ValidationError: con un mensaje por validación fallida.
    """
    failures = []
    for check in (*CHECKS, lambda n, c: check_invoice_dates(n, c, active_until)):
        try:
            check(network, config)
        except ValidationError as error:
            failures.append(str(error))
    if failures:
        raise ValidationError("\n".join(failures))
