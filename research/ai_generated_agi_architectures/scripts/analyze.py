#!/usr/bin/env python3
"""Rebuilds comparison.csv and every frequency cited in summary.md.

Deterministic: reads raw_outputs/*.md and writes comparison.csv.
No network, no API calls - pure text analysis. Run from the packet root:

    python3 scripts/analyze.py
"""
import os
import re
import csv
import glob
import collections

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "raw_outputs")

# Titles are used as delimiters instead of numbers: each model numbers its
# sections differently, but all of them write the titles.
TITULOS = [
    ("goal",         r"goal\s*(?:and|&)\s*scope|proposito y alcance"),
    ("components",   r"core\s+components|componentes\s+centrales"),
    ("dataflow",     r"data\s*flow|flujo de datos"),
    ("memory",       r"memory\s+model|modelo de memoria"),
    ("verification", r"verification\s*(?:and|&)\s*error\s*correction|verificacion"),
    ("learning",     r"learning\s+loop|bucle de aprendizaje"),
    ("failures",     r"failure\s+modes|modos de falla"),
    ("assumptions",  r"what\s+must\s+be\s+true|supuestos"),
]

SKIP = ("lyria", "openrouter-free")   # audio model / router: not architecture proposals


def limpiar(s):
    s = re.sub(r"[\*#|_`]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def secciones(txt):
    """Split by title, taking the LAST occurrence of each title (final answer)."""
    cuerpo = txt.split("---\n", 1)[-1]
    hits = {}
    for clave, pat in TITULOS:
        ps = [m.start() for m in re.finditer(pat, cuerpo, re.I)]
        if ps:
            hits[clave] = ps[-1]
    if len(hits) < 5:
        return {}
    orden = sorted(hits.items(), key=lambda kv: kv[1])
    out = {}
    for i, (clave, pos) in enumerate(orden):
        posteriores = [o for c, o in orden if c != clave and o > pos]
        fin = min(posteriores) if posteriores else len(cuerpo)
        out[clave] = cuerpo[pos:fin]
    return out


def componentes(s):
    cuerpo = re.sub(r"^[^\n]{0,90}", "", s, count=1)
    nombres = [m.group(1) for m in re.finditer(r"\*\*([A-Z][^*\n]{2,48}?)\*\*", cuerpo)]
    if len(nombres) < 3:
        nombres = [m.group(1) for m in
                   re.finditer(r"\|\s*\**([A-Z][A-Za-z0-9 \-\+/&\(\)]{2,40}?)\**\s*\|", cuerpo)]
    if len(nombres) < 3:
        nombres = [m.group(1) for m in re.finditer(
            r"(?:^|\n)[\s\-\*\u2022>]*\**([A-Z][A-Za-z0-9 \-\+&]{2,40}?)\**"
            r"\s*(?:\([A-Z]{2,6}\))?\s*[:\u2014\u2013\-]\s", cuerpo)]
    vistos, out = set(), []
    for n in nombres:
        n = n.strip(" *|_-")
        nl = n.lower()
        if (2 < len(n) < 50 and nl not in vistos
                and not re.match(r"^(the|and|one|its|this|each|core|data|memory|verif|"
                                 r"learning|failure|what|goal)\b", nl)):
            vistos.add(nl)
            out.append(n)
    return out


MARCAS_MEM = {"episodica": r"epis[oó]dic", "semantica": r"sem[aá]ntic",
              "procedural": r"procedural", "vectorial": r"\bvector",
              "grafo": r"\b(?:graph|grafo)\b",
              "de trabajo": r"working memory|buffer|working set"}

MARCAS_VER = {"desajuste de prediccion":
              r"predictive mismatch|prediction error|mismatch|deviation|divergence",
              "critico/auditor": r"\b(?:critic|auditor|monitor)\b",
              "pruebas/tests": r"\btest",
              "chequeo de consistencia": r"consisten",
              "incertidumbre/calibracion": r"uncertaint|calibrat|confidence",
              "reversion": r"rollback|revert|undo|halt"}

BLOQUES = {
    "Modelo del mundo (world model)": r"world\s*model|\bwme\b|\bwm\b|probabilistic\s*model|causal\s*model",
    "Memoria (episodica/semantica/procedural)": r"memor|episod|\bstore\b|\bkgm\b|\bcms\b|buffer|graph|index",
    "Razonamiento / planificador": r"reason|planner|\bplan\b|planning|symbolic|logic",
    "Verificacion / critico / monitor": r"verif|critic|monitor|check|audit|consisten",
    "Percepcion / encoder / grounding": r"sens|percept|encoder|grounding|interface|fusion",
    "Accion / ejecutor": r"action|execut|actuator|motor|skill",
    "Meta-cognicion / meta-aprendizaje": r"meta",
    "Atencion / control": r"attention|controller|\bac\b|scheduler|allocat",
}


def main():
    filas = []
    for f in sorted(glob.glob(os.path.join(RAW, "*.md"))):
        base = os.path.basename(f)
        if base.startswith("_"):
            continue
        modelo = base.replace(".md", "")
        if any(s in modelo for s in SKIP):
            continue
        txt = open(f).read()
        secs = secciones(txt)
        if not secs:
            print("!! no sections found: " + modelo)
            continue
        comps = componentes(secs.get("components", ""))
        mem = [k for k, p in MARCAS_MEM.items() if re.search(p, secs.get("memory", ""), re.I)]
        ver = [k for k, p in MARCAS_VER.items() if re.search(p, secs.get("verification", ""), re.I)]
        filas.append({
            "modelo": modelo, "familia": modelo.split("-")[0],
            "chars": len(txt.split("---\n", 1)[-1]),
            "n_componentes": len(comps), "componentes": "; ".join(comps[:10]),
            "memoria": ", ".join(mem) or "n/d", "verificacion": ", ".join(ver) or "n/d",
            "fallas_declaradas": limpiar(secs.get("failures", ""))[:260],
            "supuestos": limpiar(secs.get("assumptions", ""))[:260],
        })

    cols = ["modelo", "familia", "chars", "n_componentes", "componentes", "memoria",
            "verificacion", "fallas_declaradas", "supuestos"]
    with open(os.path.join(BASE, "comparison.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in filas:
            w.writerow(r)

    n = len(filas)
    freq = collections.defaultdict(set)
    for r in filas:
        for comp in [c.strip() for c in r["componentes"].split(";") if c.strip()]:
            if re.match(r"^Step \d", comp):
                continue
            for cat, pat in BLOQUES.items():
                if re.search(pat, comp, re.I):
                    freq[cat].add(r["modelo"])

    print("N = %d models\n" % n)
    print("BLOCK CONVERGENCE:")
    for cat, ms in sorted(freq.items(), key=lambda kv: -len(kv[1])):
        print("  %-45s %d/%d  (%.0f%%)" % (cat, len(ms), n, 100 * len(ms) / n))

    mem_f, ver_f = collections.Counter(), collections.Counter()
    for r in filas:
        for m in r["memoria"].split(","):
            if m.strip() and m.strip() != "n/d":
                mem_f[m.strip()] += 1
        for v in r["verificacion"].split(","):
            if v.strip() and v.strip() != "n/d":
                ver_f[v.strip()] += 1
    print("\nMEMORY TYPES:")
    for k, v in mem_f.most_common():
        print("  %-32s %d/%d" % (k, v, n))
    print("\nVERIFICATION MECHANISMS:")
    for k, v in ver_f.most_common():
        print("  %-32s %d/%d" % (k, v, n))
    print("\nwrote: comparison.csv (%d rows)" % n)


if __name__ == "__main__":
    main()
