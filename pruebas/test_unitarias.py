"""Pruebas unitarias de KuriBot (pytest). Ejecutar: python -m pytest pruebas/test_unitarias.py -v"""
from __future__ import annotations

import datetime as dt
import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kuribot import entidades as E  # noqa: E402
from kuribot import geodesia as G  # noqa: E402
from kuribot.conocimiento import BaseConocimiento  # noqa: E402
from kuribot.razonamiento import NODOS, RedAlertas, motor_alertas, primera_ventana, sensibilidad  # noqa: E402


@pytest.fixture(scope="module")
def kb():
    return BaseConocimiento()


@pytest.fixture(scope="module")
def red():
    return RedAlertas()


# ---------------------------------------------------------------- geodesia
def test_ida_y_vuelta_psad56_wgs84():
    e, n = 642600.0, 9662480.0
    we, wn = G.psad56_utm_a_wgs84_utm(e, n)
    pe, pn = G.wgs84_utm_a_psad56_utm(we, wn)
    assert abs(pe - e) < 0.01 and abs(pn - n) < 0.01


def test_desplazamiento_psad56_en_zona_del_proyecto():
    we, wn = G.psad56_utm_a_wgs84_utm(642600.0, 9662480.0)
    d = ((we - 642600) ** 2 + (wn - 9662480) ** 2) ** 0.5
    assert 400 < d < 480          # orden de magnitud conocido para Ecuador continental


def test_geograficas_a_utm():
    e, n = G.a_wgs84_utm(G.Punto(-79.7301, -3.0512, "geo", "WGS84"))
    assert abs(e - 641120.7) < 1 and abs(n - 9662664.3) < 1


def test_operacion_epsg_3990_disponible():
    assert "3990" in G.descripcion_transformacion()


# ---------------------------------------------------------------- entidades
@pytest.mark.parametrize("texto,esperado", [
    ("E 642350 N 9662120", (642350, 9662120, "utm")), ("642350, 9662120", (642350, 9662120, "utm")),
    ("x=644800 y=9661500", (644800, 9661500, "utm")), ("-3.0512, -79.7301", (-79.7301, -3.0512, "geo")),
    ("3°03'04\"S 79°43'48\"W", (-79.73, -3.0511, "geo")),
])
def test_extraer_coordenadas(texto, esperado):
    c = E.extraer_coordenada(texto)
    assert c is not None and c.punto.tipo == esperado[2]
    assert abs(c.punto.x - esperado[0]) < 1e-3 and abs(c.punto.y - esperado[1]) < 1e-3


def test_datum_detectado():
    assert E.extraer_datum("643990 9661010 psad56") == "PSAD56"
    assert E.extraer_datum("en WGS 84") == "WGS84"
    assert E.extraer_datum("sin datum") is None


def test_concesion_por_codigo_nombre_y_errata(kb):
    assert E.extraer_concesion("revisa cm 103", kb.nombres) == "CM-103"
    assert E.extraer_concesion("estado de pena blanca", kb.nombres) == "CM-101"
    assert E.extraer_concesion("la consecion el robel", kb.nombres) == "CM-104"
    assert E.extraer_concesion("una concesión cualquiera", kb.nombres) is None


def test_evidencias_con_negacion():
    ev = E.extraer_evidencias("hay maquinaria, no hay personas y el agua está turbia")
    assert ev == {"M": True, "P": False, "T": True}


def test_fechas_relativas():
    hoy = dt.date(2026, 11, 9)  # lunes
    assert E.extraer_fecha("mañana", hoy) == 1
    assert E.extraer_fecha("pasado mañana", hoy) == 2
    assert E.extraer_fecha("el viernes", hoy) == 4
    assert E.extraer_fecha("el 12 de noviembre", hoy) == 3


def test_cedula_y_enmascarado():
    assert E.cedula_valida("3000000004") and not E.cedula_valida("3000000005")
    texto, tipos = E.enmascarar_pii("cédula 3000000004, celular 0991234567, correo a@b.ec")
    assert "3000000004" not in texto and "0991234567" not in texto and set(tipos) == {"cédula", "teléfono", "correo"}


def test_emergencia_palabras_clave():
    assert E.hay_emergencia("hay gente armada en la vía")
    assert not E.hay_emergencia("se escuchan voladuras de noche")


# ---------------------------------------------------------------- red bayesiana
def test_pgmpy_coincide_con_enumeracion(red):
    for valores in itertools.product([0, 1], repeat=4):
        ev = dict(zip(["H", "A", "M", "T"], valores))
        assert abs(red.posterior(ev) - red.posterior_enumeracion(ev)) < 1e-9


def test_prior_sin_evidencia(red):
    assert abs(red.posterior({"H": 0}) - 0.10) < 1e-9 and abs(red.posterior({"H": 1}) - 0.30) < 1e-9


def test_explicacion_alternativa_reduce_posterior(red):
    ev = {"H": 0, "M": 1, "S": 1}
    assert red.posterior({**ev, "A": 1}) < 0.3 < 0.9 < red.posterior({**ev, "A": 0})


def test_monotonia_evidencias(red):
    base = {"H": 0, "A": 0}
    for nodo in [n for n in NODOS if n not in ("H", "A", "I")]:
        assert red.posterior({**base, nodo: 1}) > red.posterior(base) > red.posterior({**base, nodo: 0})


def test_sensibilidad_ordenada():
    filas = sensibilidad({"H": 0, "A": 1, "M": 1, "S": 1})
    assert filas[0]["rango"] >= filas[-1]["rango"] and filas[0]["parametro"].startswith("P(I=1")


# ---------------------------------------------------------------- reglas
def test_emergencia_prevalece():
    f = motor_alertas().inferir({"emergencia": True, "p_intrusion": 0.05, "ubicacion": "propia"})
    assert f.hechos["prioridad"] == "CRÍTICA" and f.hechos["prohibir_despliegue"] and f.disparadas[0] == "R01"


def test_tercero_no_se_califica_como_intrusion():
    f = motor_alertas().inferir({"ubicacion": "tercero", "p_intrusion": 0.95})
    assert f.hechos["prioridad"] == "REVISIÓN LEGAL" and "R21" not in f.disparadas


@pytest.mark.parametrize("p,prioridad", [(0.9, "CRÍTICA"), (0.6, "ALTA"), (0.3, "MEDIA"), (0.05, "BAJA")])
def test_umbral_prioridad(p, prioridad):
    assert motor_alertas().inferir({"ubicacion": "propia", "p_intrusion": p}).hechos["prioridad"] == prioridad


def test_prioridad_unica_y_determinista():
    import random
    from kuribot.razonamiento import REGLAS_ALERTA, MotorReglas
    hechos = {"ubicacion": "propia", "p_intrusion": 0.85, "comunidad_cercana": True, "vuelo_dias": 0}
    ref = MotorReglas(REGLAS_ALERTA).inferir(hechos).hechos
    for s in range(20):
        reglas = REGLAS_ALERTA[:]
        random.Random(s).shuffle(reglas)
        assert MotorReglas(reglas).inferir(hechos).hechos == ref


def test_supervision_humana_en_zona_sensible():
    f = motor_alertas().inferir({"ubicacion": "propia", "p_intrusion": 0.6, "area_protegida": True})
    assert f.hechos["revision_humana"] and f.hechos["aprobacion_supervisor"]


# ---------------------------------------------------------------- vuelos y base de conocimiento
def test_ventana_vuelo_evita_viento(kb):
    v = primera_ventana(1, "Norte", kb.op)
    assert v["dia"] == 2 and any("viento" in m for m in v["evaluados"][0]["motivos"])


def test_anticipacion_minima(kb):
    assert primera_ventana(0, "Sur", kb.op, urgente=False)["dia"] == 1
    assert primera_ventana(0, "Sur", kb.op, urgente=True)["dia"] == 0


def test_analisis_espacial(kb):
    assert kb.analizar_punto(642350, 9662120).ubicacion == "propia"
    assert kb.analizar_punto(642000, 9663600).ubicacion == "tercero"
    a = kb.analizar_punto(641100, 9664950)
    assert a.ubicacion == "amortiguamiento" and a.cercana == "CM-104" and abs(a.distancia_lindero_m - 100) < 1


def test_ticket_inexistente_no_se_inventa(kb):
    assert kb.ticket("GEO-2026-0999") is None and kb.ticket("GEO-2026-0412")["estado"] == "En elaboración"


def test_areas_derivadas_de_geometria(kb):
    assert kb.ficha("CM-103")["area_ha"] == 480.0


def test_riesgo_de_datum_cambia_la_concesion(kb):
    """Las mismas cifras en PSAD56 y en WGS 84 caen en concesiones distintas: el datum importa."""
    we, wn = G.psad56_utm_a_wgs84_utm(643100, 9662100)
    assert kb.analizar_punto(643100, 9662100).concesion == "CM-102"
    assert kb.analizar_punto(we, wn).concesion == "CM-101"
