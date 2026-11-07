"""Interfaz web de KuriBot (Streamlit).  Ejecutar:  streamlit run app_streamlit.py"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kuribot.conocimiento import BaseConocimiento  # noqa: E402
from kuribot.dialogo import AVISO, KuriBot  # noqa: E402
from kuribot.entidades import EVIDENCIAS  # noqa: E402
from kuribot.mapa import dibujar  # noqa: E402
from kuribot.nlu import ClasificadorIntenciones  # noqa: E402
from kuribot.razonamiento import DESCRIPCION_NODOS  # noqa: E402

st.set_page_config(page_title="KuriBot · Geomática", page_icon="🛰️", layout="wide")


@st.cache_resource(show_spinner="Cargando el modelo de lenguaje…")
def cargar_nlu():
    return ClasificadorIntenciones()


def bot() -> KuriBot:
    if "bot" not in st.session_state:
        st.session_state.bot = KuriBot(cargar_nlu(), BaseConocimiento(), registro=Path("registros/conversaciones.jsonl"))
        st.session_state.historial = []
    return st.session_state.bot


EJEMPLOS = ["Hola", "¿Cuál es el estado de la concesión La Ventana?", "¿El punto E 642600 N 9662480 PSAD56 está dentro de alguna concesión?",
            "Hay una retroexcavadora y gente extraña trabajando en Cerro Azul", "Necesito un vuelo de dron en San Gregorio mañana",
            "¿Cómo va la solicitud GEO-2026-0412?", "¿Qué EPSG uso para PSAD56 UTM 17 sur?"]

with st.sidebar:
    st.markdown("### 🛰️ KuriBot")
    st.caption(AVISO)
    st.markdown(f"**Modelo NLU:** {cargar_nlu().nombre}")
    ver_traza = st.toggle("Mostrar trazabilidad", value=True)
    st.markdown("**Ejemplos**")
    for i, e in enumerate(EJEMPLOS):
        if st.button(e, key=f"ej{i}", use_container_width=True):
            st.session_state.pendiente_ejemplo = e
    if st.button("🔄 Nueva conversación", use_container_width=True):
        for k in ("bot", "historial"):
            st.session_state.pop(k, None)
        st.rerun()
    st.caption("No escriba cédulas, teléfonos ni datos personales. Emergencias reales: ECU 911.")

st.markdown("## KuriBot · Asistente de Geomática y Catastro minero")
st.caption("Prototipo académico · datos ficticios · NLU con BETO ajustado + reglas + red bayesiana")
asistente = bot()


def mostrar_traza(r):
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown(f"**Intención:** `{r.intencion}` · **confianza** {r.confianza:.2f} · **decisión** {r.decision}")
        st.dataframe(pd.DataFrame(r.top, columns=["intención", "probabilidad"]).round(3), hide_index=True, use_container_width=True)
        if r.entidades:
            def fmt(k, v):
                if hasattr(v, "texto"):
                    return f"coordenada={v.texto.strip()}"
                if k == "evidencias":
                    return "evidencias=" + "; ".join(f"{EVIDENCIAS[e][0]} ({'sí' if b else 'no'})" for e, b in v.items())
                return f"{k}={v}"
            st.markdown("**Entidades:** " + ", ".join(fmt(k, v) for k, v in r.entidades.items() if v not in (False,)))
        if r.traza_reglas:
            st.markdown("**Reglas disparadas**")
            for t in r.traza_reglas:
                st.markdown(f"- {t}")
        if r.bayes:
            b = r.bayes
            st.markdown(f"**Red bayesiana:** P(intrusión | evidencias) = **{b['p_intrusion']:.3f}**")
            filas = [{"nodo": k, "descripción": DESCRIPCION_NODOS[k], "valor": "sí" if v else "no"} for k, v in b["evidencia"].items()]
            st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)
            if b["contribuciones"]:
                st.caption("Cambio del posterior al retirar cada evidencia: " + "; ".join(
                    f"{EVIDENCIAS[k][0]} {d:+.3f}" for k, d in b["contribuciones"]))
        if r.vuelo and r.vuelo.get("evaluados"):
            st.markdown("**Evaluación de ventanas de vuelo**")
            st.dataframe(pd.DataFrame([{"día": e["dia"], "apto": e["apto"], "viento m/s": e["viento"], "lluvia mm/h": e["lluvia"],
                                        "visib. km": e["visibilidad"], "motivos": "; ".join(e["motivos"])} for e in r.vuelo["evaluados"]]),
                         hide_index=True, use_container_width=True)
        st.caption(f"Latencia total {r.latencia_ms:.0f} ms (NLU {r.latencia_nlu_ms:.0f} ms)")
    with c2:
        if r.mapa and (r.mapa.get("concesion") or r.mapa.get("punto")):
            st.pyplot(dibujar(asistente.kb, r.mapa.get("concesion"), r.mapa.get("punto")), use_container_width=True)


for msg in st.session_state.historial:
    with st.chat_message(msg["rol"], avatar="🧑‍💼" if msg["rol"] == "user" else "🛰️"):
        st.markdown(msg["texto"])
        if msg.get("r") is not None and ver_traza:
            with st.expander("Trazabilidad de la respuesta", expanded=msg.get("ultimo", False)):
                mostrar_traza(msg["r"])

entrada = st.chat_input("Escriba su consulta…") or st.session_state.pop("pendiente_ejemplo", None)
if entrada:
    for m in st.session_state.historial:
        m["ultimo"] = False
    r = asistente.responder(entrada)
    st.session_state.historial.append({"rol": "user", "texto": entrada})
    texto = r.texto + ("\n\n" + "\n".join(f"⚠️ {a}" for a in r.avisos) if r.avisos else "")
    st.session_state.historial.append({"rol": "assistant", "texto": texto, "r": r, "ultimo": True})
    st.rerun()
