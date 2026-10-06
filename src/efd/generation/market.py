"""Facturación legítima: conexión preferencial, matriz intersectorial, montos, IVA y estacionalidad.

``MarketModel`` también expone los muestreadores de contrapartes, montos y fechas que usan los patrones
de fraude y el camuflaje, de modo que el comercio simulado por las fantasmas sigue las mismas reglas
que el comercio legítimo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from efd.generation.config import GeneratorConfig

ONE_DAY = np.timedelta64(1, "D")


def invoice_block(
    issuer: np.ndarray, receiver: np.ndarray, issue_date: np.ndarray, subtotal: np.ndarray
) -> pd.DataFrame:
    """Bloque interno de facturas: posiciones de emisor y receptor, fecha y subtotal sin impuestos."""
    return pd.DataFrame(
        {
            "issuer": np.asarray(issuer, dtype=np.int64),
            "receiver": np.asarray(receiver, dtype=np.int64),
            "issue_date": np.asarray(issue_date, dtype="datetime64[D]"),
            "subtotal": np.asarray(subtotal, dtype=np.float64),
        }
    )


def round_to_multiples(
    rng: np.random.Generator, amounts: np.ndarray, multiples: tuple[int, ...]
) -> np.ndarray:
    """Redondea cada monto a un múltiplo elegido al azar entre ``multiples`` (al menos un múltiplo)."""
    multiple = rng.choice(np.asarray(multiples, dtype=np.float64), size=len(amounts))
    return np.maximum(multiple, np.round(np.asarray(amounts) / multiple) * multiple)


def _sample_rows(rng: np.random.Generator, probabilities: np.ndarray) -> np.ndarray:
    """Muestrea una columna por fila de una matriz de probabilidades (filas normalizadas)."""
    cumulative = probabilities.cumsum(axis=1)
    draws = rng.random(len(probabilities))[:, None] * cumulative[:, -1:]
    return np.minimum((draws >= cumulative).sum(axis=1), probabilities.shape[1] - 1)


class MarketModel:
    """Modelo del comercio legítimo entre contribuyentes."""

    def __init__(self, config: GeneratorConfig, taxpayers: pd.DataFrame) -> None:
        """
        Args:
            config: configuración del escenario.
            taxpayers: contribuyentes con índice posicional, tal como los crea ``build_population``.
        """
        self.config = config
        market = config.market
        self.sectors = tuple(config.population.sectors)
        sizes = config.population.sizes
        self.sector = pd.Categorical(taxpayers["sector"], categories=self.sectors).codes.astype(np.int64)
        self.size = pd.Categorical(taxpayers["size"], categories=sizes).codes.astype(np.int64)
        self.start = taxpayers["start_date"].to_numpy().astype("datetime64[D]")
        self.legit = taxpayers["is_shell"].to_numpy() == 0
        self.n = len(taxpayers)

        matrix = np.array(
            [[market.sector_matrix[buyer].get(supplier, 0.0) for supplier in self.sectors]
             for buyer in self.sectors],
            dtype=np.float64,
        )
        self.matrix = matrix / matrix.sum(axis=1, keepdims=True)
        self.attractiveness = np.array([market.attractiveness[s] for s in sizes], dtype=np.int64)
        self.extra_suppliers = np.array([market.extra_suppliers_mean[s] for s in sizes])
        self.purchase_weight = np.array([market.purchase_weight[s] for s in sizes])
        self.log_mean = np.array([market.amount_log_mean[s] for s in sizes])
        self.customer_counts = np.zeros(self.n, dtype=np.int64)

        self.period_start = np.datetime64(config.period.start, "D")
        self.period_end = np.datetime64(config.period.end, "D")
        days = np.arange(self.period_start, self.period_end + ONE_DAY)
        months = days.astype("datetime64[M]")
        next_months = (months + np.timedelta64(1, "M")).astype("datetime64[D]")
        days_in_month = (next_months - months.astype("datetime64[D]")).astype(np.int64)
        seasonality = np.asarray(market.monthly_seasonality)[months.astype(np.int64) % 12]
        self._day_cdf = np.concatenate([[0.0], np.cumsum(seasonality / days_in_month)])

    # Muestreadores compartidos con patrones y camuflaje.

    def sample_dates(
        self, rng: np.random.Generator, lower: np.ndarray, upper: np.ndarray
    ) -> np.ndarray:
        """Fechas con estacionalidad mensual, cada una dentro de ``[lower, upper]`` (inclusive).

        Raises:
            ValueError: si algún intervalo está vacío o fuera del período.
        """
        low = ((np.asarray(lower, dtype="datetime64[D]") - self.period_start) // ONE_DAY).astype(np.int64)
        high = ((np.asarray(upper, dtype="datetime64[D]") - self.period_start) // ONE_DAY).astype(np.int64)
        if low.size == 0:
            return np.empty(0, dtype="datetime64[D]")
        if (low > high).any() or low.min() < 0 or high.max() >= self._day_cdf.size - 1:
            raise ValueError("Intervalo de fechas vacío o fuera del período simulado.")
        cdf = self._day_cdf
        draws = cdf[low] + rng.random(low.size) * (cdf[high + 1] - cdf[low])
        day = np.clip(np.searchsorted(cdf, draws, side="right") - 1, low, high)
        return self.period_start + day.astype("timedelta64[D]")

    def sample_amounts(
        self, rng: np.random.Generator, buyers: np.ndarray, round_fraction: float | None = None
    ) -> np.ndarray:
        """Subtotales lognormales según el tamaño del comprador.

        Una fracción ``round_fraction`` (por defecto, la de la configuración) se redondea a montos
        redondos, como ocurre también en el comercio legítimo.
        """
        market = self.config.market
        buyers = np.asarray(buyers, dtype=np.int64)
        amounts = np.round(
            np.exp(self.log_mean[self.size[buyers]] + market.amount_log_sigma * rng.standard_normal(buyers.size)),
            2,
        )
        amounts = np.maximum(amounts, 0.01)
        fraction = market.round_amount_fraction if round_fraction is None else round_fraction
        is_round = rng.random(buyers.size) < fraction
        amounts[is_round] = round_to_multiples(rng, amounts[is_round], market.round_multiples)
        return amounts

    def choose_counterparts(
        self,
        rng: np.random.Generator,
        n: int,
        sector: int,
        latest_start: np.datetime64,
        role: str,
        replace: bool,
    ) -> np.ndarray:
        """Elige contrapartes legítimas para un contribuyente del sector ``sector``.

        Args:
            n: número de contrapartes (sin reemplazo se limita a las disponibles).
            sector: código de sector del contribuyente que busca contrapartes.
            latest_start: solo se eligen contribuyentes que iniciaron hasta esta fecha.
            role: ``"buyer"`` si la contraparte le compra (peso por tamaño de compra y matriz
                intersectorial) o ``"supplier"`` si le vende (peso por clientes, atractivo y matriz).
            replace: si una contraparte puede repetirse.
        """
        candidates = np.flatnonzero(self.legit & (self.start <= latest_start))
        if role == "buyer":
            weights = self.purchase_weight[self.size[candidates]] * self.matrix[self.sector[candidates], sector]
        elif role == "supplier":
            weights = (
                self.customer_counts[candidates] + self.attractiveness[self.size[candidates]]
            ) * self.matrix[sector, self.sector[candidates]]
        else:
            raise ValueError(f"Rol desconocido: '{role}'.")
        if weights.sum() <= 0:
            weights = np.ones(candidates.size)
        if not replace:
            n = min(n, int((weights > 0).sum()))
        if n <= 0 or candidates.size == 0:
            return np.empty(0, dtype=np.int64)
        return rng.choice(candidates, size=n, replace=replace, p=weights / weights.sum())

    def split_vat(self, rng: np.random.Generator, subtotal: np.ndarray) -> pd.DataFrame:
        """Separa cada subtotal en tarifa 0 % y tarifa general y calcula IVA y total."""
        market = self.config.market
        subtotal = np.asarray(subtotal, dtype=np.float64)
        has_zero_rated = rng.random(subtotal.size) < market.zero_rated_fraction
        share = np.where(has_zero_rated, rng.random(subtotal.size), 0.0)
        subtotal_0 = np.round(subtotal * share, 2)
        subtotal_15 = np.round(subtotal - subtotal_0, 2)
        vat = np.round(market.vat_rate * subtotal_15, 2)
        return pd.DataFrame(
            {
                "subtotal_0": subtotal_0,
                "subtotal_15": subtotal_15,
                "vat": vat,
                "total": np.round(subtotal_0 + subtotal_15 + vat, 2),
            }
        )

    # Comercio legítimo.

    def build_legitimate_invoices(
        self, rng: np.random.Generator, representatives: pd.DataFrame, taxpayer_ids: np.ndarray
    ) -> pd.DataFrame:
        """Genera todas las facturas entre contribuyentes legítimos.

        Args:
            representatives: representantes con identificador entero, para el comercio intragrupo.
            taxpayer_ids: identificador de cada posición, para unir los representantes.
        """
        suppliers, buyers = self._preferential_relations(rng)
        extra = [
            self._reciprocal_relations(rng, suppliers, buyers),
            self._group_relations(rng, representatives, taxpayer_ids),
            self._cycle_relations(rng),
        ]
        suppliers = np.concatenate([suppliers, *(pair[0] for pair in extra)])
        buyers = np.concatenate([buyers, *(pair[1] for pair in extra)])
        keep = suppliers != buyers
        keys = np.unique(suppliers[keep] * self.n + buyers[keep])
        suppliers, buyers = keys // self.n, keys % self.n

        counts = self._invoice_counts(rng, buyers)
        issuer = np.repeat(suppliers, counts)
        receiver = np.repeat(buyers, counts)
        lower = np.maximum(np.maximum(self.start[issuer], self.start[receiver]), self.period_start)
        dates = self.sample_dates(rng, lower, np.full(issuer.size, self.period_end))
        return invoice_block(issuer, receiver, dates, self.sample_amounts(rng, receiver))

    def _preferential_relations(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Relaciones proveedor → cliente por conexión preferencial dentro de cada sector.

        Los contribuyentes llegan en orden de inicio de actividades. Cada uno elige ``1 + Poisson``
        proveedores entre los que ya existen: primero el sector del proveedor, con la fila de la
        matriz intersectorial de su propio sector, y luego el proveedor dentro de ese sector. Con
        probabilidad ``uniform_choice_probability`` la elección es proporcional al atractivo inicial
        según tamaño; en otro caso, proporcional al número de clientes ya conseguidos. Cada lista
        contiene a un proveedor tantas veces como su peso, así que tomar un elemento al azar equivale
        a muestrear de forma proporcional.
        """
        market = self.config.market
        legit = np.flatnonzero(self.legit)
        order = legit[np.lexsort((rng.random(legit.size), self.start[legit]))]
        picks = 1 + rng.poisson(self.extra_suppliers[self.size[order]])
        buyer_of_pick = np.repeat(order, picks)
        sector_of_pick = _sample_rows(rng, self.matrix[self.sector[buyer_of_pick]])
        use_attractiveness = rng.random(buyer_of_pick.size) < market.uniform_choice_probability
        position = rng.random(buyer_of_pick.size)

        by_attractiveness: list[list[int]] = [[] for _ in self.sectors]
        by_customers: list[list[int]] = [[] for _ in self.sectors]
        supplier_of_pick = np.full(buyer_of_pick.size, -1, dtype=np.int64)
        pick = 0
        for buyer, n_picks in zip(order.tolist(), picks.tolist()):
            chosen: set[int] = set()
            for j in range(pick, pick + n_picks):
                sector = sector_of_pick[j]
                pool = by_customers[sector]
                if use_attractiveness[j] or not pool:
                    pool = by_attractiveness[sector]
                if not pool:
                    continue
                supplier = pool[int(position[j] * len(pool))]
                if supplier not in chosen:
                    chosen.add(supplier)
                    supplier_of_pick[j] = supplier
                    by_customers[sector].append(supplier)
            pick += n_picks
            by_attractiveness[self.sector[buyer]].extend([buyer] * int(self.attractiveness[self.size[buyer]]))

        found = supplier_of_pick >= 0
        suppliers, buyers = supplier_of_pick[found], buyer_of_pick[found]
        self.customer_counts += np.bincount(suppliers, minlength=self.n)
        return suppliers, buyers

    def _reciprocal_relations(
        self, rng: np.random.Generator, suppliers: np.ndarray, buyers: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Comercio recíproco: una fracción de relaciones también ocurre en sentido inverso."""
        chosen = rng.random(suppliers.size) < self.config.market.reciprocity_fraction
        return buyers[chosen], suppliers[chosen]

    def _group_relations(
        self, rng: np.random.Generator, representatives: pd.DataFrame, taxpayer_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Comercio entre sociedades legítimas de un mismo grupo empresarial."""
        position = pd.Series(np.arange(self.n), index=taxpayer_ids)
        members = representatives.assign(position=position.loc[representatives["taxpayer_id"]].to_numpy())
        members = members[self.legit[members["position"].to_numpy()]]
        group_sizes = members.groupby("representative_id")["position"].transform("size")
        members = members[group_sizes > 1]
        pairs = members.merge(members, on="representative_id", suffixes=("_supplier", "_buyer"))
        pairs = pairs[pairs["position_supplier"] != pairs["position_buyer"]]
        chosen = rng.random(len(pairs)) < self.config.market.group_trade_probability
        return (
            pairs["position_supplier"].to_numpy()[chosen],
            pairs["position_buyer"].to_numpy()[chosen],
        )

    def _cycle_relations(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Ciclos legítimos de comercio de 3 a 6 empresas, para que un ciclo no implique fraude."""
        market = self.config.market
        legit = np.flatnonzero(self.legit)
        n_cycles = round(market.cycles_per_1000 * legit.size / 1000)
        low, high = market.cycle_length
        suppliers, buyers = [], []
        for _ in range(n_cycles):
            members = rng.choice(legit, size=int(rng.integers(low, high + 1)), replace=False)
            suppliers.append(members)
            buyers.append(np.roll(members, -1))
        if not suppliers:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return np.concatenate(suppliers), np.concatenate(buyers)

    def _invoice_counts(self, rng: np.random.Generator, buyers: np.ndarray) -> np.ndarray:
        """Facturas por relación: al menos una, y el resto del total repartido por multinomial.

        El peso de cada relación combina el tamaño del comprador y una intensidad lognormal propia de
        la relación.
        """
        market = self.config.market
        target = round(market.invoices_per_taxpayer * int(self.legit.sum()))
        counts = np.ones(buyers.size, dtype=np.int64)
        remaining = target - buyers.size
        if remaining > 0:
            weights = self.purchase_weight[self.size[buyers]] * rng.lognormal(
                0.0, market.relation_intensity_sigma, buyers.size
            )
            counts += rng.multinomial(remaining, weights / weights.sum())
        return counts
