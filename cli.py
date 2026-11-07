"""Interfaz de consola de KuriBot.  Ejecutar:  python cli.py   (escriba «salir» para terminar)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kuribot.dialogo import AVISO, KuriBot  # noqa: E402


def main():
    bot = KuriBot(registro=Path("registros/consola.jsonl"))
    print(f"KuriBot · {bot.nlu.nombre}\n{AVISO}\nEscriba «salir» para terminar.\n")
    while True:
        try:
            texto = input("Usted › ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not sys.stdin.isatty():
            print(texto)  # eco de la entrada cuando la consola se alimenta desde un archivo
        if texto.lower() in {"salir", "exit", "quit"}:
            break
        if not texto:
            continue
        r = bot.responder(texto)
        print(f"KuriBot › {r.texto}")
        for a in r.avisos:
            print(f"   ⚠ {a}")
        extra = f" · P(intrusión)={r.bayes['p_intrusion']:.3f}" if r.bayes else ""
        print(f"   [intención={r.intencion} conf={r.confianza:.2f} decisión={r.decision} reglas={','.join(r.reglas) or '—'}"
              f"{extra} · {r.latencia_ms:.0f} ms]\n")


if __name__ == "__main__":
    main()
