"""P4 — Representante compartido."""

from __future__ import annotations

import numpy as np

from efd.generation.config import check_int_range
from efd.generation.patterns.base import FraudPattern, GenerationState, draw_int
from efd.generation.patterns.registry import register_pattern


@register_pattern
class SharedRepresentativePattern(FraudPattern):
    """Grupo de sociedades fantasmas con el mismo representante legal y clientes en común.

    Los montos siguen la distribución legítima y los miembros compran insumos a proveedores
    legítimos, de modo que la señal es relacional (representante y beneficiarios compartidos) y no
    tabular.
    """

    name = "shared_representative"

    def __init__(
        self,
        group_size: list[int],
        shared_beneficiaries: list[int],
        beneficiaries_per_member: list[int],
        invoices_per_beneficiary: list[int],
        suppliers_per_member: list[int],
        invoices_per_supplier: list[int],
    ) -> None:
        """
        Args:
            group_size: rango de fantasmas por grupo.
            shared_beneficiaries: rango del tamaño del grupo común de beneficiarios.
            beneficiaries_per_member: rango de beneficiarios del grupo común a los que vende cada miembro.
            invoices_per_beneficiary: rango de facturas a cada beneficiario.
            suppliers_per_member: rango de proveedores legítimos de cada miembro.
            invoices_per_supplier: rango de facturas recibidas de cada proveedor.
        """
        self._group_size = tuple(group_size)
        self.shared_beneficiaries = tuple(shared_beneficiaries)
        self.beneficiaries_per_member = tuple(beneficiaries_per_member)
        self.invoices_per_beneficiary = tuple(invoices_per_beneficiary)
        self.suppliers_per_member = tuple(suppliers_per_member)
        self.invoices_per_supplier = tuple(invoices_per_supplier)
        check_int_range(self._group_size, "group_size", minimum=2)
        check_int_range(self.shared_beneficiaries, "shared_beneficiaries", minimum=1)
        check_int_range(self.beneficiaries_per_member, "beneficiaries_per_member", minimum=1)
        check_int_range(self.invoices_per_beneficiary, "invoices_per_beneficiary", minimum=1)
        check_int_range(self.suppliers_per_member, "suppliers_per_member")
        check_int_range(self.invoices_per_supplier, "invoices_per_supplier", minimum=1)

    @property
    def group_size(self) -> tuple[int, int]:
        return self._group_size

    def inject(self, state: GenerationState, rng: np.random.Generator) -> None:
        market = state.market
        for members in self.groups(state, rng):
            state.share_representative(members)
            _, upper = state.window(members)
            pool = market.choose_counterparts(
                rng,
                draw_int(rng, self.shared_beneficiaries),
                market.sector[members[0]],
                upper,
                "buyer",
                replace=False,
            )
            for member in members:
                n_buyers = min(draw_int(rng, self.beneficiaries_per_member), pool.size)
                buyers = rng.choice(pool, size=n_buyers, replace=False)
                receivers = np.repeat(buyers, draw_int(rng, self.invoices_per_beneficiary, size=buyers.size))
                state.add_invoices(
                    "fraud",
                    member,
                    receivers,
                    state.counterpart_dates(rng, member, receivers),
                    market.sample_amounts(rng, receivers),
                )

                suppliers = market.choose_counterparts(
                    rng,
                    draw_int(rng, self.suppliers_per_member),
                    market.sector[member],
                    upper,
                    "supplier",
                    replace=False,
                )
                issuers = np.repeat(suppliers, draw_int(rng, self.invoices_per_supplier, size=suppliers.size))
                state.add_invoices(
                    "fraud",
                    issuers,
                    member,
                    state.counterpart_dates(rng, member, issuers),
                    market.sample_amounts(rng, np.full(issuers.size, member)),
                )
