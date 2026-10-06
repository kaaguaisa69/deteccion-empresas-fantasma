"""Genera un escenario: ``python -m efd.generation --config configs/generator.yaml``."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from efd.generation.config import load_generator_config
from efd.generation.generator import InvoiceNetworkGenerator


def main(argv: list[str] | None = None) -> Path:
    """Lee la configuración, genera el escenario y lo escribe; devuelve su carpeta."""
    parser = argparse.ArgumentParser(description="Genera un escenario de la red sintética de facturación.")
    parser.add_argument("--config", type=Path, required=True, help="archivo YAML del generador")
    parser.add_argument(
        "--output-root", type=Path, default=None, help="carpeta raíz de salida (por defecto, la del YAML)"
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    directory = InvoiceNetworkGenerator(load_generator_config(args.config)).run(args.output_root)
    print(f"Escenario escrito en {directory} en {time.perf_counter() - started:.1f} s")
    return directory


if __name__ == "__main__":
    main()
