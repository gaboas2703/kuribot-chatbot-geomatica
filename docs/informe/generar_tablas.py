"""Genera las tablas y cifras del informe directamente desde resultados/*.json (sin transcripción manual)."""
from __future__ import annotations

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
RES = RAIZ / "resultados"


def c(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def cargar(n):
    return json.loads((RES / n).read_text(encoding="utf-8"))


def tablas() -> dict[str, str]:
    v1, v2 = cargar("nlp_final_v1.json"), cargar("nlp_final_v2.json")
    a1, a2 = cargar("aceptacion_v1.json"), cargar("aceptacion_v2.json")
    lat_m2 = json.loads((RES / "latencia_m2_v1.json").read_text()) if (RES / "latencia_m2_v1.json").exists() else None
    T = {}
    cv = v1["validacion_cruzada"]
    filas = ["| Modelo | F1 macro VC (media ± DE) | F1 prueba limpia [IC 95 %] | F1 prueba con ruido [IC 95 %] | ECE calibrado (limpia) | Parámetros entrenables | Latencia p95 (ms) |",
             "|:--|--:|--:|--:|--:|--:|--:|"]
    for m, r in v1["modelos"].items():
        pl, pr = r["prueba"]["prueba_limpia"], r["prueba"]["prueba_ruido"]
        lat = r["latencia"]["p95_ms"] if m != "M2" or lat_m2 is None else lat_m2["p95_ms"]
        filas.append(f"| {r['nombre']} | {c(cv[m]['f1_macro_media'])} ± {c(cv[m]['f1_macro_de'])} | {c(pl['f1_macro'])} "
                     f"[{c(pl['f1_macro_ic95_bootstrap'][0], 2)}–{c(pl['f1_macro_ic95_bootstrap'][1], 2)}] | {c(pr['f1_macro'])} "
                     f"[{c(pr['f1_macro_ic95_bootstrap'][0], 2)}–{c(pr['f1_macro_ic95_bootstrap'][1], 2)}] | {c(pl['ece_calibrado'])} | "
                     + f"{r['parametros_entrenables']:,}".replace(",", " ") + f" | {c(lat, 1)} |")
    T["TABLA_MODELOS"] = "\n".join(filas)

    filas = ["| Versión | Modelo | F1 limpia | F1 ruido | Exactitud limpia | ECE calibrado | Épocas entrenadas | τ |",
             "|:--|:--|--:|--:|--:|--:|--:|--:|"]
    for ver, r in [("v1", v1), ("v2", v2)]:
        for m in ("M0", "M3"):
            x = r["modelos"][m]
            pl, pr = x["prueba"]["prueba_limpia"], x["prueba"]["prueba_ruido"]
            ep = len(x["info"].get("historia", [])) or "—"
            tau = x.get("umbral", {}).get("tau_elegido_en_validacion")
            filas.append(f"| {ver} | {x['nombre']} | {c(pl['f1_macro'])} | {c(pr['f1_macro'])} | {c(pl['exactitud'])} | "
                         f"{c(pl['ece_calibrado'])} | {ep} | {c(tau, 2) if tau is not None else '—'} |")
    T["TABLA_VERSIONES"] = "\n".join(filas)

    etq = v2["etiquetas"]
    rl, rr = v2["modelos"]["M3"]["prueba"]["prueba_limpia"]["reporte"], v2["modelos"]["M3"]["prueba"]["prueba_ruido"]["reporte"]
    filas = ["| Intención | Precisión | Exhaustividad | F1 (limpia) | F1 (ruido) | Soporte |", "|:--|--:|--:|--:|--:|--:|"]
    for e in etq:
        filas.append(f"| {e.replace('_', ' ')} | {c(rl[e]['precision'])} | {c(rl[e]['recall'])} | {c(rl[e]['f1-score'])} | "
                     f"{c(rr[e]['f1-score'])} | {int(rl[e]['support'])} |")
    filas.append(f"| **Promedio macro** | **{c(rl['macro avg']['precision'])}** | **{c(rl['macro avg']['recall'])}** | "
                 f"**{c(rl['macro avg']['f1-score'])}** | **{c(rr['macro avg']['f1-score'])}** | 252 |")
    T["TABLA_CLASES"] = "\n".join(filas)

    filas = ["| Comparación | Conjunto | Solo M3 acierta | Solo el otro acierta | p (McNemar exacta) |", "|:--|:--|--:|--:|--:|"]
    for ver, r in (("v1", v1), ("v2", v2)):
        for k, d in r["mcnemar"].items():
            otro = k.split("_vs_")[1].split("_prueba")[0]
            conj = "limpia" if "limpia" in k else "ruido"
            filas.append(f"| M3 vs {otro} ({ver}) | {conj} | {d['solo_A_acierta']} | {d['solo_B_acierta']} | {c(d['p_valor'], 3)} |")
    T["TABLA_MCNEMAR"] = "\n".join(filas)

    idx1 = {e["id"]: e for e in a1["escenarios"]}
    filas = ["| ID | Escenario | Última entrada | Intención | Acción / reglas | v1 | v2 | ms |", "|:--|:--|:--|:--|:--|:-:|:-:|--:|"]
    for e in a2["escenarios"]:
        u = e["turnos"][-1]
        ent = u["entrada"] if len(u["entrada"]) <= 48 else u["entrada"][:46] + "…"
        acc = (u["accion"] or u["decision"] or "—").replace("_", " ")
        acc = acc if len(acc) <= 34 else acc[:32] + "…"
        reglas = ", ".join(r for r in u["reglas"] if r.startswith("R"))[:28]
        filas.append(f"| {e['id']} | {e['descripcion']} | {ent.replace('|', '/')} | {u['intencion'].replace('_', ' ')} | {acc}{(' · ' + reglas) if reglas else ''} | "
                     f"{'Sí' if idx1[e['id']]['aprobado'] else 'No'} | {'Sí' if e['aprobado'] else 'No'} | {c(u['latencia_ms'], 0)} |")
    T["TABLA_ACEPTACION"] = "\n".join(filas)

    bayes = []
    for e in a2["escenarios"]:
        for u in e["turnos"]:
            if u.get("p_intrusion") is not None:
                prio = next((p for p in ("CRÍTICA", "ALTA", "MEDIA", "BAJA", "REVISIÓN LEGAL") if f"prioridad {p}" in u["respuesta"]), "—")
                bayes.append(f"| {e['id']} | {u['entrada'][:60]} | {c(u['p_intrusion'])} | {prio} | {', '.join(u['reglas'])} |")
    T["TABLA_BAYES"] = "| ID | Reporte | P(I \\| e) | Prioridad | Reglas disparadas |\n|:--|:--|--:|:--|:--|\n" + "\n".join(bayes)

    s1, s2 = a1["resumen"], a2["resumen"]
    T["ACEPT_V1"] = f"{s1['aprobados']}/{s1['escenarios']}"
    T["ACEPT_V2"] = f"{s2['aprobados']}/{s2['escenarios']}"
    T["ACEPT_TASA_V1"] = c(100 * s1["tasa"], 1)
    T["ACEPT_TASA_V2"] = c(100 * s2["tasa"], 1)
    T["ACEPT_LAT_P95"] = c(s2["latencia_p95_ms"], 1)
    T["ACEPT_LAT_MEDIA"] = c(s2["latencia_media_ms"], 1)
    T["TURNOS"] = str(s2["turnos"])
    m3v2 = v2["modelos"]["M3"]
    T["F1L"] = c(m3v2["prueba"]["prueba_limpia"]["f1_macro"])
    T["F1R"] = c(m3v2["prueba"]["prueba_ruido"]["f1_macro"])
    return T


if __name__ == "__main__":
    for k, v in tablas().items():
        print(f"== {k}\n{v}\n")
