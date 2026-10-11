"""
Point d'entree unique du projet GNN.

Lance toutes les experiences, ou une selection, depuis la racine du depot.
Chaque modele / experience reste aussi executable separement via son script.

Usage (racine du depot) :
    python main.py                  # tout lancer
    python main.py --list           # lister les etapes
    python main.py --only gat gcn   # selection
    python main.py --skip comparison evaluate_gcn
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Ordre d'execution par defaut (du plus leger au plus long)
PIPELINE: list[tuple[str, Path, str]] = [
    (
        "baselines",
        ROOT / "experiments" / "run_baselines.py",
        "Baselines liens (Distance / Jaccard / Adamic-Adar)",
    ),
    (
        "comparison",
        ROOT / "experiments" / "run_comparison.py",
        "GAE vs VGAE (anomalies de liens, multi-seeds)",
    ),
    (
        "ablation",
        ROOT / "experiments" / "run_ablation.py",
        "Ablation GAE (taux 10 %)",
    ),
    (
        "top10",
        ROOT / "experiments" / "run_top10_suspicious.py",
        "Top 10 liens suspects (GAE + VGAE)",
    ),
    (
        "gcn",
        ROOT / "src" / "gcn.py",
        "DOMINANT / GCN (anomalies de noeuds, graphe original)",
    ),
    (
        "evaluate_gcn",
        ROOT / "src" / "evaluate_gcn.py",
        "DOMINANT / GCN (evaluation avec anomalies injectees)",
    ),
    (
        "gat",
        ROOT / "experiments" / "run_gat.py",
        "GAT (regression population + top 20 suspects)",
    ),
]

ALIASES = {
    "gae": "comparison",
    "vgae": "comparison",
    "dominant": "gcn",
    "links": "comparison",
    "nodes_gcn": "gcn",
    "nodes_gat": "gat",
}


def _resolve_names(names: list[str] | None) -> list[str]:
    valid = {key for key, _, _ in PIPELINE}
    if not names:
        return [key for key, _, _ in PIPELINE]

    resolved: list[str] = []
    for name in names:
        key = ALIASES.get(name, name)
        if key not in valid:
            raise SystemExit(
                f"Etape inconnue: {name!r}. "
                f"Valides: {', '.join(sorted(valid))} "
                f"(alias: {', '.join(sorted(ALIASES))})"
            )
        if key not in resolved:
            resolved.append(key)
    return resolved


def _run_step(key: str, script: Path, description: str) -> int:
    print("\n" + "=" * 70)
    print(f"[{key}] {description}")
    print(f"    -> {script.relative_to(ROOT)}")
    print("=" * 70)
    if not script.is_file():
        print(f"ERREUR: fichier introuvable: {script}")
        return 1
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=False,
    )
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lance les experiences GNN (tous les modeles ou une selection).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Affiche les etapes disponibles et quitte.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="STEP",
        help="Execute uniquement ces etapes (ex: --only gat gcn).",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        metavar="STEP",
        help="Execute tout sauf ces etapes.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue les etapes suivantes si une etape echoue.",
    )
    args = parser.parse_args(argv)

    if args.list:
        print("Etapes disponibles (lancer depuis la racine du depot):\n")
        for key, script, description in PIPELINE:
            print(f"  {key:14s}  {script.relative_to(ROOT)}")
            print(f"                 {description}")
        print("\nAlias:", ", ".join(f"{a}->{t}" for a, t in sorted(ALIASES.items())))
        print("\nExemples:")
        print("  python main.py")
        print("  python main.py --only comparison gat")
        print("  python main.py --skip evaluate_gcn")
        return 0

    if args.only and args.skip:
        raise SystemExit("Utiliser --only OU --skip, pas les deux.")

    selected = _resolve_names(args.only)
    if args.skip:
        to_skip = set(_resolve_names(args.skip))
        selected = [k for k in selected if k not in to_skip]

    by_key = {key: (script, desc) for key, script, desc in PIPELINE}
    # Conserver l'ordre du pipeline
    ordered = [key for key, _, _ in PIPELINE if key in selected]

    print("Projet GNN — orchestration des experiences")
    print(f"Racine : {ROOT}")
    print(f"Etapes : {', '.join(ordered)}")

    failed: list[str] = []
    for key in ordered:
        script, description = by_key[key]
        code = _run_step(key, script, description)
        if code != 0:
            failed.append(key)
            print(f"\n[{key}] ECHEC (code={code})")
            if not args.continue_on_error:
                print("Arret. Relancer avec --continue-on-error pour poursuivre.")
                return code

    print("\n" + "=" * 70)
    if failed:
        print(f"Termine avec echecs: {', '.join(failed)}")
        return 1
    print("Toutes les etapes selectionnees se sont terminees correctement.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
