"""Ensambla el informe con el formato del equipo (plantilla Word de la Semana 3) y genera DOCX y PDF.

Flujo: Markdown (texto propio) -> citas APA y numeración de tablas/figuras -> Pandoc (DOCX con los estilos de la
plantilla) -> posprocesado (portada UEES, encabezado, tablas APA, ecuaciones numeradas, código del anexo, índice)
-> LibreOffice (PDF). El índice se calcula en dos pasadas con la paginación real del PDF, de modo que Word lo
muestre ya completo al abrir el archivo, sin pedir actualizar campos.

Uso: python docs/informe/construir_informe.py
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.packuri import PackURI
from docx.opc.part import Part
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Cm, Pt, RGBColor

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
PLANTILLA = AQUI / "plantilla"
sys.path.insert(0, str(AQUI))
from referencias import CITAS, REFERENCIAS  # noqa: E402

PARTES = ["portada.md", "parte1.md", "parte2.md", "parte3.md", "_referencias.md", "anexos.md"]
PATRON = re.compile(r"\[((?:-?@[\w]+)(?:\s*;\s*-?@[\w]+)*)\]")
URL_REPO = "https://github.com/gaboas2703/kuribot-chatbot-geomatica"
ENCABEZADO = ("Inteligencia Artificial · Semana 4", "Proyecto integral: ChatBot KuriBot")
ANCHO_TEXTO = 9072  # twips (A4 con márgenes de 2,5 cm)

CODIGO = [
    ("kuribot/entidades.py", "Extracción de entidades y datos personales"),
    ("kuribot/geodesia.py", "Transformaciones geodésicas y utilidades UTM"),
    ("kuribot/conocimiento.py", "Base de conocimiento: catastro, operación, FAQ y tickets"),
    ("kuribot/nlu_modelos.py", "Modelos de intenciones M0–M3, calibración y métricas"),
    ("kuribot/nlu.py", "Inferencia de intenciones y decisión aceptar, aclarar o abstenerse"),
    ("kuribot/razonamiento.py", "Motor de reglas, red bayesiana y ventanas de vuelo"),
    ("kuribot/dialogo.py", "Gestor de diálogo y generación de respuestas"),
    ("kuribot/mapa.py", "Mapa estático del catastro"),
    ("app_streamlit.py", "Interfaz web"),
    ("cli.py", "Interfaz de consola"),
    ("datos/construir_base.py", "Base de conocimiento sintética"),
    ("datos/construir_corpus.py", "Corpus de intenciones y particiones"),
    ("experimentos/entrenar_nlp.py", "Entrenamiento, validación cruzada y evaluación"),
    ("experimentos/complementos.py", "Latencia sin caché y McNemar del modelo final"),
    ("pruebas/test_unitarias.py", "Pruebas unitarias"),
    ("pruebas/suite_aceptacion.py", "Suite de aceptación y confirmación"),
    ("scripts/extraer_beto.py", "Extracción de BETO desde el pipeline de spaCy"),
]


# ----------------------------------------------------------------------------- Markdown
def citar(m: re.Match, usados: set) -> str:
    claves = [c.strip() for c in m.group(1).split(";")]
    partes, solo_anio = [], all(c.startswith("-@") for c in claves)
    for c in claves:
        k = c.lstrip("-@")
        if k not in CITAS:
            raise SystemExit(f"Cita sin definir: {k}")
        usados.add(k)
        autor, anio = CITAS[k]
        partes.append(anio if c.startswith("-@") else f"{autor}, {anio}")
    if not solo_anio:
        partes.sort(key=lambda s: s.lower())
    return "(" + "; ".join(partes) + ")"


def ensamblar() -> str:
    usados: set = set()
    textos = ["@@REFERENCIAS@@" if p == "_referencias.md" else (AQUI / p).read_text(encoding="utf-8") for p in PARTES]
    md = "\n\n".join(textos)
    md = md.replace("@@RESUMEN@@", (AQUI / "resumen.md").read_text(encoding="utf-8"))
    md = PATRON.sub(lambda m: citar(m, usados), md)
    sin_citar = sorted(set(REFERENCIAS) - usados)
    if sin_citar:
        raise SystemExit(f"Referencias sin citar en el texto: {sin_citar}")
    clave = lambda k: re.sub(r"[^a-z]", "", REFERENCIAS[k].lower().translate(str.maketrans("ñøáéíóú", "noaeiou")))[:60]  # noqa: E731
    orden = sorted(usados, key=lambda k: clave(k) + REFERENCIAS[k][:200])
    refs = ('# Referencias\n\n::: {custom-style="Referencia APA"}\n' + "\n\n".join(REFERENCIAS[k] for k in orden) + "\n:::\n")
    md = md.replace("@@REFERENCIAS@@", refs).replace("@@NREF@@", str(len(orden)))
    return md


def numerar(md: str) -> str:
    """Convierte <<tabla:id|título>>, <<figura:id|ruta|ancho|título>> y <<nota|texto>> al estilo APA del formato."""
    num, cont = {}, {"tabla": 0, "figura": 0}
    for tipo, ident in re.findall(r"<<(tabla|figura):([\w-]+)\|", md):
        cont[tipo] += 1
        num[f"{tipo}:{ident}"] = cont[tipo]

    def leyenda(texto):
        return f'\n::: {{custom-style="Leyenda de tabla o figura"}}\n{texto}\n:::\n'

    def tabla(m):
        n = num[f"tabla:{m.group(1)}"]
        return leyenda(f"**Tabla {n}.**") + leyenda(f"*{m.group(2).strip()}*") + "\n"

    def figura(m):
        n = num[f"figura:{m.group(1)}"]
        return (leyenda(f"**Figura {n}.**") + leyenda(f"*{m.group(4).strip()}*")
                + f'\n::: {{custom-style="Figura"}}\n![]({m.group(2)}){{width={m.group(3)}%}}\n:::\n')

    md = re.sub(r"<<tabla:([\w-]+)\|(.+?)>>", tabla, md)
    md = re.sub(r"<<figura:([\w-]+)\|(.+?)\|(\d+)\|(.+?)>>", figura, md)
    md = re.sub(r"\n?<<nota\|(.+?)>>", lambda m: f'\n\n::: {{custom-style="Nota de tabla"}}\n*Nota.* {m.group(1).strip()}\n:::\n', md)

    def ref(m):
        k = f"{m.group(1)}:{m.group(2)}"
        if k not in num:
            raise SystemExit(f"Referencia sin destino: {k}")
        return f"{'Tabla' if m.group(1) == 'tabla' else 'Figura'} {num[k]}"

    return re.sub(r"\{\{(tabla|figura):([\w-]+)\}\}", ref, md)


def rellenar(md: str) -> str:
    from generar_tablas import tablas
    t = tablas()
    v2 = json.loads((RAIZ / "resultados" / "nlp_final_v2.json").read_text(encoding="utf-8"))
    m3 = v2["modelos"]["M3"]
    pl = m3["prueba"]["prueba_limpia"]
    c = lambda x, d=3: f"{x:.{d}f}".replace(".", ",")  # noqa: E731
    t["FICHA_MODELO"] = ("BETO (bert-base-spanish-wwm-cased, versión del pipeline es_dep_news_trf 3.8.0 de spaCy) ajustado de extremo a "
                         "extremo, con media de la última capa y capa lineal; " + c(m3["parametros_entrenables"] / 1e6, 1)
                         + " millones de parámetros (intenciones_beto_v2, float16, 219 MB)")
    t["FICHA_DATOS"] = (f"Corpus sintético en español: {v2['particion_final']['entrenamiento'] // 2} frases de entrenamiento más una copia "
                        f"con ruido por frase ({v2['particion_final']['entrenamiento']} en total) y {v2['particion_final']['validacion']} de validación, "
                        "de 414 plantillas y 14 intenciones")
    t["FICHA_METRICAS"] = (f"F1 macro {c(pl['f1_macro'])} (limpia) y {c(m3['prueba']['prueba_ruido']['f1_macro'])} (ruido); exactitud "
                           f"{c(pl['exactitud'])} [IC Wilson {c(pl['exactitud_ic95_wilson'][0], 2)}–{c(pl['exactitud_ic95_wilson'][1], 2)}]; "
                           f"ECE {c(pl['ece_calibrado'])}; latencia p95 {c(m3['latencia']['p95_ms'], 1)} ms en CPU")
    t["FICHA_T"] = c(m3["temperatura"], 2)
    filas = ["| Ejecución | Modelo NLU | Escenarios | Aprobados | Tasa | Latencia media (ms) | Latencia p95 (ms) |", "|:--|:--|--:|--:|--:|--:|--:|"]
    for nombre, arch in [("Aceptación, iteración 1", "aceptacion_v1.json"), ("Aceptación, iteración 2", "aceptacion_v2.json"),
                         ("Confirmación (escenarios nuevos)", "confirmacion_v2.json")]:
        r = json.loads((RAIZ / "resultados" / arch).read_text(encoding="utf-8"))["resumen"]
        filas.append(f"| {nombre} | {r['modelo']} | {r['escenarios']} | {r['aprobados']} | {c(100 * r['tasa'], 1)} % | "
                     f"{c(r['latencia_media_ms'], 1)} | {c(r['latencia_p95_ms'], 1)} |")
    t["ACEPTACION_RESUMEN"] = "\n".join(filas)
    marca = lambda s: f"\n```{{=openxml}}\n<w:p><w:r><w:t>{s}</w:t></w:r></w:p>\n```\n"  # noqa: E731
    t["PYTEST"], t["CODIGO_ANEXO"] = marca("§PYTEST§"), marca("§CODIGO§")
    dibujo = (PLANTILLA / "logo_drawing.xml").read_text(encoding="utf-8")
    dibujo = re.sub(r' wp14:\w+="[^"]*"', "", dibujo)
    dibujo = re.sub(r"<wp14:sizeRel[HV].*?</wp14:sizeRel[HV]>", "", dibujo, flags=re.S)
    t["LOGO"] = dibujo
    for k, v in t.items():
        md = md.replace(f"@@{k}@@", v)
    resto = re.findall(r"@@[A-Z0-9_]+@@", md)
    if resto:
        raise SystemExit(f"Marcadores sin reemplazar: {sorted(set(resto))}")
    return md


# ----------------------------------------------------------------------------- plantilla
def preparar_referencia(ruta: Path):
    """Plantilla de Pandoc: estilos, encabezado y pie del formato del equipo, con títulos APA en negro."""
    d = Document(str(PLANTILLA / "formato_uees.docx"))
    body = d.element.body
    for el in list(body):
        if el.tag != qn("w:sectPr"):
            body.remove(el)
    st = d.styles

    def ppr(s):
        return s.element.get_or_add_pPr()

    def jc(s, val):
        p = ppr(s)
        e = p.find(qn("w:jc"))
        if e is None:
            e = OxmlElement("w:jc"); p.append(e)
        e.set(qn("w:val"), val)

    for nombre, tam, centro in [("Heading 1", 12, True), ("Heading 2", 11, False), ("Heading 3", 11, False), ("Título del índice", 12, True)]:
        s = st[nombre]
        s.font.size, s.font.bold = Pt(tam), True
        s.font.color.rgb = RGBColor(0, 0, 0)
        s.paragraph_format.keep_with_next = True
        jc(s, "center" if centro else "left")
    st["Heading 3"].paragraph_format.space_before = Pt(8)
    st["Heading 3"].paragraph_format.space_after = Pt(3)
    ley = st["Leyenda de tabla o figura"].paragraph_format
    ley.keep_with_next, ley.keep_together = True, True

    def nuevo(nombre, tipo=WD_STYLE_TYPE.PARAGRAPH, base="Normal"):
        try:
            return st[nombre]
        except KeyError:
            s = st.add_style(nombre, tipo)
            if base:
                s.base_style = st[base]
            return s

    nuevo("Body Text"); nuevo("First Paragraph")
    comp = nuevo("Compact").paragraph_format
    comp.first_line_indent, comp.space_after = Cm(0), Pt(3)
    fig = nuevo("Figura").paragraph_format
    fig.first_line_indent, fig.space_before, fig.space_after = Cm(0), Pt(6), Pt(3)
    fig.alignment = WD_ALIGN_PARAGRAPH.CENTER
    vb = nuevo("Verbatim Char", WD_STYLE_TYPE.CHARACTER, None)
    vb.font.name, vb.font.size = "Consolas", Pt(9.5)
    for nombre, izq in [("toc 1", 0), ("toc 2", 0.39)]:
        f = st[nombre].paragraph_format
        f.first_line_indent, f.left_indent, f.space_after = Cm(0), Cm(izq), Pt(2)
    d.save(str(ruta))


# ----------------------------------------------------------------------------- posprocesado
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _el(tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(k), str(v))
    return e


ORDEN_PPR = ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr", "suppressLineNumbers",
             "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap", "overflowPunct", "topLinePunct", "autoSpaceDE",
             "autoSpaceDN", "bidi", "adjustRightInd", "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents",
             "suppressOverlap", "jc", "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle",
             "rPr", "sectPr", "pPrChange"]
ORDEN_RPR = ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike", "dstrike", "outline", "shadow", "emboss",
             "imprint", "noProof", "snapToGrid", "vanish", "webHidden", "color", "spacing", "w", "kern", "position", "sz", "szCs",
             "highlight", "u", "effect", "bdr", "shd", "fitText", "vertAlign", "rtl", "cs", "em", "lang", "eastAsianLayout",
             "specVanish", "oMath"]


def poner(padre, hijo, orden):
    """Inserta (o reemplaza) un hijo respetando el orden del esquema OOXML."""
    nombre = hijo.tag.split("}")[1]
    for e in padre.findall(hijo.tag):
        padre.remove(e)
    pos = orden.index(nombre)
    for i, e in enumerate(padre):
        n = e.tag.split("}")[1] if "}" in e.tag else e.tag
        if n in orden and orden.index(n) > pos:
            padre.insert(i, hijo)
            return hijo
    padre.append(hijo)
    return hijo


def _ppr(p):
    ppr = p.find(qn("w:pPr"))
    if ppr is None:
        ppr = _el("w:pPr"); p.insert(0, ppr)
    return ppr


def _p_codigo(texto: str):
    p = _el("w:p")
    ppr = _el("w:pPr"); ppr.append(_el("w:pStyle", **{"w:val": "Cdigo"})); p.append(ppr)
    if texto:
        r = _el("w:r"); t = _el("w:t"); t.text = texto
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve"); r.append(t); p.append(r)
    return p


def _p_titulo(texto: str, nivel: int, marcador: str | None = None):
    p = _el("w:p")
    ppr = _el("w:pPr"); ppr.append(_el("w:pStyle", **{"w:val": f"Ttulo{nivel}"})); p.append(ppr)
    r = _el("w:r"); t = _el("w:t"); t.text = texto
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve"); r.append(t); p.append(r)
    return p


def _reemplazar_marca(body, marca: str, nuevos: list):
    for p in body.iter(qn("w:p")):
        if "".join(t.text or "" for t in p.iter(qn("w:t"))) == marca:
            padre, i = p.getparent(), p.getparent().index(p)
            padre.remove(p)
            for k, n in enumerate(nuevos):
                padre.insert(i + k, n)
            return
    raise SystemExit(f"No se encontró la marca {marca}")


def anexo_codigo(body):
    nuevos = []
    for k, (ruta, desc) in enumerate(CODIGO, 1):
        nuevos.append(_p_titulo(f"D.{k} {desc} ({ruta})", 3))
        for linea in (RAIZ / ruta).read_text(encoding="utf-8").rstrip().split("\n"):
            nuevos.append(_p_codigo(linea.replace("\t", "    ")))
    _reemplazar_marca(body, "§CODIGO§", nuevos)
    salida = (RAIZ / "resultados" / "pruebas_unitarias.txt").read_text(encoding="utf-8").strip().split("\n")
    _reemplazar_marca(body, "§PYTEST§", [_p_codigo(l) for l in salida])


def _borde(lado, val="single", sz=4):
    return _el(f"w:{lado}", **{"w:val": val, "w:sz": sz, "w:space": 0, "w:color": "000000" if val != "nil" else "auto"})


def formatear_tablas(doc):
    """Tablas APA: línea gruesa arriba y abajo, fina bajo el encabezado, sin verticales, 9 pt y anchos por contenido."""
    for t in doc.tables:
        tbl = t._tbl
        filas = tbl.findall(qn("w:tr"))
        if not filas:
            continue
        textos_f1 = "".join(x.text or "" for x in filas[0].iter(qn("w:t")))
        if textos_f1.startswith("Integrantes"):
            continue  # tabla de la portada
        if tbl.find(".//" + qn("m:oMathPara")) is not None:
            continue
        ncol = max(len(f.findall(qn("w:tc"))) for f in filas)
        # anchos por contenido: deseado según el percentil 90 del largo del texto; mínimo, la palabra más larga
        minimos, deseados = [], []
        for k in range(ncol):
            textos = []
            for f in filas:
                celdas = f.findall(qn("w:tc"))
                if k < len(celdas):
                    textos.append("".join(x.text or "" for x in celdas[k].iter(qn("w:t"))))
            largos = sorted(len(x) for x in textos)
            p90 = largos[int(0.9 * (len(largos) - 1))] if largos else 4
            palabra = max((len(w) for x in textos for w in x.split()), default=3)
            minimos.append(min(palabra, 18) * 100 + 180)
            deseados.append(max(min(p90, 48) * 100 + 180, minimos[-1]))
        if sum(deseados) <= ANCHO_TEXTO:
            anchos = [int(d * ANCHO_TEXTO / sum(deseados)) for d in deseados]
        elif sum(minimos) >= ANCHO_TEXTO:
            anchos = [int(m * ANCHO_TEXTO / sum(minimos)) for m in minimos]
        else:
            libre, extra = ANCHO_TEXTO - sum(minimos), [d - m for d, m in zip(deseados, minimos)]
            anchos = [int(m + libre * e / max(1, sum(extra))) for m, e in zip(minimos, extra)]
        tblPr = tbl.tblPr
        for hijo in list(tblPr):
            tblPr.remove(hijo)
        tblPr.append(_el("w:tblW", **{"w:w": ANCHO_TEXTO, "w:type": "dxa"}))
        tblPr.append(_el("w:jc", **{"w:val": "center"}))
        bordes = _el("w:tblBorders")
        for lado in ("top", "left", "bottom", "right", "insideH", "insideV"):
            bordes.append(_borde(lado, "nil", 0))
        tblPr.append(bordes)
        tblPr.append(_el("w:tblLayout", **{"w:type": "fixed"}))
        mar = _el("w:tblCellMar")
        for lado, v in (("top", 30), ("left", 70), ("bottom", 30), ("right", 70)):
            mar.append(_el(f"w:{lado}", **{"w:w": v, "w:type": "dxa"}))
        tblPr.append(mar)
        grid = tbl.find(qn("w:tblGrid"))
        for gc in list(grid):
            grid.remove(gc)
        for a in anchos:
            grid.append(_el("w:gridCol", **{"w:w": a}))
        for i, f in enumerate(filas):
            trPr = f.find(qn("w:trPr"))
            if trPr is None:
                trPr = _el("w:trPr"); f.insert(0, trPr)
            for hijo in list(trPr):
                trPr.remove(hijo)
            trPr.append(_el("w:cantSplit"))
            if i == 0:
                trPr.append(_el("w:tblHeader"))
            for k, tc in enumerate(f.findall(qn("w:tc"))):
                tcPr = tc.find(qn("w:tcPr"))
                if tcPr is None:
                    tcPr = _el("w:tcPr"); tc.insert(0, tcPr)
                span = tcPr.find(qn("w:gridSpan"))
                for hijo in list(tcPr):
                    tcPr.remove(hijo)
                tcPr.append(_el("w:tcW", **{"w:w": anchos[k] if k < len(anchos) else 1000, "w:type": "dxa"}))
                if span is not None:
                    tcPr.append(span)
                b = _el("w:tcBorders")
                b.append(_borde("top", "single", 12) if i == 0 else _borde("top", "nil", 0))
                if i == 0:
                    b.append(_borde("bottom", "single", 4))
                elif i == len(filas) - 1:
                    b.append(_borde("bottom", "single", 12))
                tcPr.append(b)
                tcPr.append(_el("w:vAlign", **{"w:val": "center" if i else "bottom"}))
                for p in tc.findall(qn("w:p")):
                    ppr = _ppr(p)
                    for e in ppr.findall(qn("w:pStyle")):
                        ppr.remove(e)
                    poner(ppr, _el("w:spacing", **{"w:before": 0, "w:after": 0, "w:line": 240, "w:lineRule": "auto"}), ORDEN_PPR)
                    poner(ppr, _el("w:ind", **{"w:left": 0, "w:firstLine": 0}), ORDEN_PPR)
                    jc = ppr.find(qn("w:jc"))
                    val = {"start": "left", "end": "right"}.get(jc.get(qn("w:val")) if jc is not None else "left", jc.get(qn("w:val")) if jc is not None else "left")
                    poner(ppr, _el("w:jc", **{"w:val": "left" if val == "both" else val}), ORDEN_PPR)
                    if i == 0:
                        poner(ppr, _el("w:keepNext"), ORDEN_PPR)
                    for r in p.iter(qn("w:r")):
                        rpr = r.find(qn("w:rPr"))
                        if rpr is None:
                            rpr = _el("w:rPr"); r.insert(0, rpr)
                        poner(rpr, _el("w:sz", **{"w:val": 18}), ORDEN_RPR)
                        poner(rpr, _el("w:szCs", **{"w:val": 18}), ORDEN_RPR)
                        if i == 0:
                            poner(rpr, _el("w:b"), ORDEN_RPR)


def espacio_tras_tablas(body):
    """Separa del cuerpo de texto el párrafo que sigue a una tabla sin nota."""
    hijos = list(body)
    for i, e in enumerate(hijos[:-1]):
        sig = hijos[i + 1]
        if e.tag == qn("w:tbl") and sig.tag == qn("w:p"):
            ps = sig.find(qn("w:pPr") + "/" + qn("w:pStyle"))
            if ps is None or ps.get(qn("w:val")) in ("BodyText", "FirstParagraph", "Compact"):
                poner(_ppr(sig), _el("w:spacing", **{"w:before": 160}), ORDEN_PPR)


def numerar_ecuaciones(body):
    n = 0
    for p in list(body.iter(qn("w:p"))):
        if p.find(qn("m:oMathPara")) is None or p.getparent().tag != qn("w:body"):
            continue
        n += 1
        tbl = parse_xml(
            f'<w:tbl {nsdecls("w")}><w:tblPr><w:tblW w:w="{ANCHO_TEXTO}" w:type="dxa"/><w:tblBorders>'
            + "".join(f'<w:{l} w:val="nil"/>' for l in ("top", "left", "bottom", "right", "insideH", "insideV"))
            + '</w:tblBorders><w:tblLayout w:type="fixed"/></w:tblPr><w:tblGrid><w:gridCol w:w="907"/>'
            f'<w:gridCol w:w="{ANCHO_TEXTO - 1814}"/><w:gridCol w:w="907"/></w:tblGrid><w:tr><w:trPr><w:cantSplit/></w:trPr>'
            '<w:tc><w:tcPr><w:tcW w:w="907" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr><w:p/></w:tc>'
            f'<w:tc><w:tcPr><w:tcW w:w="{ANCHO_TEXTO - 1814}" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr></w:tc>'
            '<w:tc><w:tcPr><w:tcW w:w="907" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr><w:p><w:pPr><w:ind w:firstLine="0"/>'
            f'<w:jc w:val="right"/></w:pPr><w:r><w:t>({n})</w:t></w:r></w:p></w:tc></w:tr></w:tbl>')
        padre, i = p.getparent(), p.getparent().index(p)
        padre.remove(p)
        ppr = _ppr(p)
        for hijo in list(ppr):
            ppr.remove(hijo)
        ppr.append(_el("w:spacing", **{"w:before": 60, "w:after": 60}))
        ppr.append(_el("w:ind", **{"w:firstLine": 0}))
        ppr.append(_el("w:jc", **{"w:val": "center"}))
        tbl.findall(qn("w:tr"))[0].findall(qn("w:tc"))[1].append(p)
        padre.insert(i, tbl)


def portada_y_secciones(doc):
    part = doc.part
    # logo UEES (EMF) en la portada
    blob = (PLANTILLA / "logo_uees.emf").read_bytes()
    img = Part(PackURI("/word/media/logo_uees.emf"), "image/x-emf", blob, part.package)
    rid = part.relate_to(img, RT.IMAGE)
    for blip in doc.element.body.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}blip"):
        clave = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
        if blip.get(clave) == "rIdLOGOUEES":
            blip.set(clave, rid)
    # sección del cuerpo: encabezado, pie y numeración desde 1
    rels = {r.target_ref: r.rId for r in part.rels.values() if not r.is_external}
    sect = doc.element.body.find(qn("w:sectPr"))
    for hijo in list(sect):
        if hijo.tag in (qn("w:headerReference"), qn("w:footerReference"), qn("w:pgNumType")):
            sect.remove(hijo)
    hdr = _el("w:headerReference", **{"w:type": "default"}); hdr.set(qn("r:id"), rels["header1.xml"])
    ftr = _el("w:footerReference", **{"w:type": "default"}); ftr.set(qn("r:id"), rels["footer1.xml"])
    sect.insert(0, ftr); sect.insert(0, hdr)
    pg = _el("w:pgNumType", **{"w:start": 1})
    sect.insert(list(sect).index(sect.find(qn("w:pgMar"))) + 1, pg)
    # textos del encabezado
    for rel in part.rels.values():
        if rel.reltype == RT.HEADER:
            h = rel.target_part
            for t in h.element.iter(qn("w:t")):
                if t.text and "Semana 3" in t.text:
                    t.text = ENCABEZADO[0]
                elif t.text and "Laboratorio comparativo" in t.text:
                    t.text = ENCABEZADO[1]


def estilos_cuerpo(doc):
    """Párrafos de Pandoc con el aspecto del formato: cuerpo como Normal, listas justificadas sin sangría."""
    for p in doc.element.body.iter(qn("w:p")):
        ppr = p.find(qn("w:pPr"))
        ps = ppr.find(qn("w:pStyle")) if ppr is not None else None
        val = ps.get(qn("w:val")) if ps is not None else None
        if val in ("BodyText", "FirstParagraph"):
            ppr.remove(ps)
        if ppr is not None and ppr.find(qn("w:numPr")) is not None:
            poner(ppr, _el("w:spacing", **{"w:after": 60}), ORDEN_PPR)
            poner(ppr, _el("w:jc", **{"w:val": "both"}), ORDEN_PPR)


def indice(doc, paginas: dict | None):
    """Índice como campo TOC con resultado ya calculado (sin aviso de actualización al abrir en Word)."""
    entradas = []
    for p in doc.element.body.iter(qn("w:p")):
        ppr = p.find(qn("w:pPr"))
        ps = ppr.find(qn("w:pStyle")) if ppr is not None else None
        val = ps.get(qn("w:val")) if ps is not None else None
        if val in ("Ttulo1", "Ttulo2"):
            texto = "".join(t.text or "" for t in p.iter(qn("w:t"))).strip()
            marca = next((b.get(qn("w:name")) for b in p.iter(qn("w:bookmarkStart"))), None)
            if marca is None:
                marca = f"_Toc{len(entradas):04d}"
                p.insert(1, _el("w:bookmarkStart", **{"w:id": 9000 + len(entradas), "w:name": marca}))
                p.append(_el("w:bookmarkEnd", **{"w:id": 9000 + len(entradas)}))
            entradas.append((1 if val == "Ttulo1" else 2, texto, marca))
    nuevos = [parse_xml(f'<w:p {nsdecls("w")}><w:pPr><w:pStyle w:val="Ttulodelndice"/><w:pageBreakBefore/></w:pPr><w:r><w:t>Índice</w:t></w:r></w:p>')]
    for k, (nivel, texto, marca) in enumerate(entradas):
        pag = (paginas or {}).get(k, "00")
        inicio = ('<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> TOC \\o "1-2" \\h \\z \\u </w:instrText></w:r>'
                  '<w:r><w:fldChar w:fldCharType="separate"/></w:r>') if k == 0 else ""
        fin = '<w:r><w:fldChar w:fldCharType="end"/></w:r>' if k == len(entradas) - 1 else ""
        texto_xml = texto.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        nuevos.append(parse_xml(
            f'<w:p {nsdecls("w", "r")}><w:pPr><w:pStyle w:val="TDC{nivel}"/><w:tabs><w:tab w:val="right" w:leader="dot" w:pos="{ANCHO_TEXTO}"/></w:tabs></w:pPr>'
            f'{inicio}<w:hyperlink w:anchor="{marca}" w:history="1"><w:r><w:t xml:space="preserve">{texto_xml}</w:t></w:r>'
            f'<w:r><w:tab/></w:r><w:r><w:t>{pag}</w:t></w:r></w:hyperlink>{fin}</w:p>'))
    _reemplazar_marca(doc.element.body, "§TOC§", nuevos)
    # el cuerpo, las referencias (APA) y los anexos empiezan en página nueva
    for p in doc.element.body.iter(qn("w:p")):
        ps = p.find(qn("w:pPr") + "/" + qn("w:pStyle"))
        if ps is not None and ps.get(qn("w:val")) == "Ttulo1" and "".join(t.text or "" for t in p.iter(qn("w:t"))).strip() in ("Introducción", "Referencias", "Anexos"):
            poner(_ppr(p), _el("w:pageBreakBefore"), ORDEN_PPR)
    return entradas


def ordenar_propiedades(body):
    """Ordena los hijos de pPr y rPr según el esquema OOXML (Word rechaza el orden incorrecto)."""
    for tag, orden in (("w:pPr", ORDEN_PPR), ("w:rPr", ORDEN_RPR)):
        for e in body.iter(qn(tag)):
            hijos = [c for c in e if isinstance(c.tag, str)]
            nombres = [c.tag.split("}")[1] for c in hijos]
            if all(n in orden for n in nombres):
                for c in sorted(hijos, key=lambda c: orden.index(c.tag.split("}")[1])):
                    e.remove(c); e.append(c)


def propiedades(doc):
    cp = doc.core_properties
    cp.title = "KuriBot: ChatBot inteligente con deep learning para la mesa de servicios de Geomática"
    cp.author = "Gabriel Ayala Sánchez; Daniel Castro M.; Ricardo Ruiz L."
    cp.subject = "Proyecto integral de inteligencia artificial · UEES · Semana 4"
    cp.keywords = "ChatBot; BETO; red bayesiana; motor de reglas; geomática; minería"
    cp.comments = "Informe del proyecto (versión editable)"
    cp.last_modified_by = "Gabriel Ayala"
    s = doc.settings.element
    for e in s.findall(qn("w:updateFields")):
        s.remove(e)


def construir_docx(md_path: Path, salida: Path, paginas: dict | None):
    ref = AQUI / "plantilla" / "_referencia.docx"
    preparar_referencia(ref)
    subprocess.run(["pandoc", str(md_path), "-o", str(salida), "--reference-doc", str(ref),
                    "--from", "markdown+pipe_tables+fenced_divs+raw_attribute+tex_math_dollars-implicit_figures",
                    "--resource-path", f"{AQUI}:{AQUI.parent / 'figuras'}:{AQUI.parent / 'capturas'}"], cwd=AQUI, check=True)
    doc = Document(str(salida))
    body = doc.element.body
    portada_y_secciones(doc)
    estilos_cuerpo(doc)
    numerar_ecuaciones(body)
    formatear_tablas(doc)
    espacio_tras_tablas(body)
    anexo_codigo(body)
    entradas = indice(doc, paginas)
    ordenar_propiedades(body)
    ordenar_propiedades(doc.styles.element)
    propiedades(doc)
    doc.save(str(salida))
    return entradas


# ----------------------------------------------------------------------------- PDF e índice
def a_pdf(docx: Path) -> Path:
    perfil = Path(tempfile.gettempdir()) / "lo_perfil_kuribot"
    subprocess.run(["soffice", f"-env:UserInstallation=file://{perfil}", "--headless", "--convert-to", "pdf",
                    "--outdir", str(docx.parent), str(docx)], check=True, capture_output=True, timeout=600)
    return docx.with_suffix(".pdf")


def paginas_de_titulos(pdf: Path, entradas: list) -> dict:
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout).group(1))
    lineas_por_pag = []
    for k in range(1, n + 1):
        txt = subprocess.run(["pdftotext", "-f", str(k), "-l", str(k), str(pdf), "-"], capture_output=True, text=True).stdout
        lineas_por_pag.append([" ".join(l.split()) for l in txt.split("\n") if l.strip()])
    norm = lambda s: " ".join(s.split())  # noqa: E731
    # el cuerpo empieza después de la última página del índice (líneas con puntos guía)
    inicio = max(k for k, ls in enumerate(lineas_por_pag) if any("....." in l for l in ls)) + 1
    res, pag = {}, inicio
    for i, (nivel, texto, _) in enumerate(entradas):
        objetivo = norm(texto)
        if objetivo in ("Resumen",):
            res[i] = str(next(k for k, ls in enumerate(lineas_por_pag) if "Resumen" in ls))  # página física - 1 (portada sin número)
            continue
        encontrado = None
        for k in range(pag, n):
            ls = lineas_por_pag[k]
            for j, l in enumerate(ls):
                junto = l + (" " + ls[j + 1] if j + 1 < len(ls) else "")
                if l == objetivo or (len(l) >= 25 and objetivo.startswith(l) and junto.startswith(objetivo[:len(junto)])):
                    encontrado = k
                    break
            if encontrado is not None:
                break
        if encontrado is None:
            raise SystemExit(f"No se encontró el título en el PDF: {texto}")
        pag = encontrado
        res[i] = str(encontrado)  # el número impreso = página física - 1
    return res


def main():
    md = numerar(ensamblar())
    md = rellenar(md)
    (AQUI / "informe.md").write_text(md, encoding="utf-8")
    salida = AQUI / "informe.docx"
    entradas = construir_docx(AQUI / "informe.md", salida, None)
    pdf = a_pdf(salida)
    paginas = paginas_de_titulos(pdf, entradas)
    construir_docx(AQUI / "informe.md", salida, paginas)
    pdf = a_pdf(salida)
    comprobacion = paginas_de_titulos(pdf, entradas)
    if comprobacion != paginas:
        print("Aviso: la paginación cambió entre pasadas", {k: (paginas[k], comprobacion[k]) for k in paginas if paginas[k] != comprobacion[k]})
    print(f"DOCX y PDF generados: {salida.name}, {pdf.name}")
    print("Índice:", [(e[1][:40], paginas[i]) for i, e in enumerate(entradas)])


if __name__ == "__main__":
    main()
