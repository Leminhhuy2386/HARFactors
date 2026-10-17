"""Figures 2 and 3 of the manuscript as an editable PowerPoint deck, one figure per slide.

Every element is a native PowerPoint shape (rectangles, rounded boxes, text boxes, arrows, hatched cells), so the
figures can be edited in PowerPoint. The deck is written with the standard library only (OOXML in a zip; no
python-pptx), from the same inputs as the PDF figures:

- slide 1, study design: results/figure_data/fig03_design.json (runs, cells by run set, UCI-HAR test groups), as drawn
  by src/visualization/detail_figures.py (fig03);
- slide 2, the shared skeleton of the networks (fig05): a fixed dataflow, as in the manuscript.

A preview PNG of each slide is drawn from the same shape list with matplotlib, to check the layout where no
PowerPoint renderer is available (its font is wider than Arial, so the preview errs towards overflow).

    uv run --no-sync python -m src.visualization.pptx_export
        -> figures/fig02_fig03.pptx, figures/fig02_fig03_slide1.png, figures/fig02_fig03_slide2.png
"""
from __future__ import annotations

import json
import time
import zipfile
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from src.data.sets import S1, S2, S3  # noqa: E402
from src.utils import paths as PATHS  # noqa: E402
from src.visualization.detail_figures import EXCLUDED, NET  # noqa: E402

DESIGN = PATHS.RESULTS / "figure_data" / "fig03_design.json"
SENSORS = PATHS.DOCS / "sources" / "arena_table3_sensors.json"
OUT = PATHS.FIGURES / "fig02_fig03.pptx"
W_IN, H_IN = 13.333, 7.5  # 16:9 widescreen
EMU = 914400
FONT = "Arial"
# the validated categorical slots and neutrals of src/visualization/letter_figures.py, without '#'
BLUE, ORANGE, GREEN, YELLOW = "256ABF", "D95926", "199E70", "C98500"
INK, INK2, GRID, LIGHT, WHITE = "0B0B0B", "3D3C39", "DEDCD5", "ECEBE8", "FFFFFF"


def tint(hex_colour: str, f: float) -> str:
    """hex_colour mixed with white: f = 1 is the colour, f = 0 white."""
    rgb = [int(hex_colour[i:i + 2], 16) for i in (0, 2, 4)]
    return "".join(f"{round(255 - f * (255 - c)):02X}" for c in rgb)


# ------------------------------------------------------------------------------------------------------ shapes
def box(x, y, w, h, fill=None, line=None, lw=1.0, round_=False, pattern=None, name="shape"):
    return {"kind": "box", "x": x, "y": y, "w": w, "h": h, "fill": fill, "line": line, "lw": lw, "round": round_,
            "pattern": pattern, "name": name, "paras": []}


def text(x, y, w, h, paras, align="l", anchor="t", vert=False, name="text"):
    """paras: list of runs-lists; a run is (string, size_pt, bold, italic, colour)."""
    return {"kind": "text", "x": x, "y": y, "w": w, "h": h, "paras": paras, "align": align, "anchor": anchor,
            "vert": vert, "name": name}


def labelled(shape, paras, align="ctr", anchor="ctr"):
    shape |= {"paras": paras, "align": align, "anchor": anchor, "vert": False}
    return shape


def line(x0, y0, x1, y1, colour=INK, lw=1.25, arrow=False, name="line"):
    return {"kind": "line", "x0": x0, "y0": y0, "x1": x1, "y1": y1, "colour": colour, "lw": lw, "arrow": arrow,
            "name": name}


def run(s, size, bold=False, italic=False, colour=INK):
    return (s, size, bold, italic, colour)


# ------------------------------------------------------------------------------------------------------ slide 1
def slide_design() -> list[dict]:
    d = json.loads(DESIGN.read_text())
    names = json.loads(SENSORS.read_text())["datasets"]
    runs = d["runs"]
    cells = {(c["network"], c["dataset"]): c for c in d["cells"]}
    shapes = [text(0.5, 0.3, 12.3, 0.6, [[run("Study design: runs analysed and person-disjoint splits", 24,
                                              True)]], name="title")]
    # legend of the run sets
    legend = [(BLUE, None, f"S1 development, {runs['S1']:,} runs"), (ORANGE, None, f"S2 held-out (H1), {runs['S2']:,} runs"),
              (GREEN, None, f"S3 benchmark networks, {runs['S3']:,} runs"),
              (WHITE, INK2, "excluded by the inclusion rule")]
    x = 0.5
    for fill, hatch, lab in legend:
        shapes.append(box(x, 1.12, 0.2, 0.2, fill=fill if hatch is None else None, line=None if hatch is None else INK2,
                          lw=0.75, pattern=None if hatch is None else (INK2, WHITE), name="legend key"))
        shapes.append(text(x + 0.27, 1.04, 2.85, 0.36, [[run(lab, 11)]], anchor="ctr", name="legend label"))
        x += 3.1

    # (a) networks x datasets
    dev, held = list(S1["datasets"]), list(S2["datasets"])
    pitch, size, gap_c, gap_r = 0.28, 0.24, 0.22, 0.16
    x0, y0 = 2.75, 2.42
    xs = {dd: x0 + j * pitch for j, dd in enumerate(dev)} | {dd: x0 + len(dev) * pitch + gap_c + j * pitch
                                                             for j, dd in enumerate(held)}
    rows = S1["models"] + ["whar17"]
    ys = {m: y0 + i * pitch for i, m in enumerate(S1["models"])} | {"whar17": y0 + len(S1["models"]) * pitch + gap_r}
    colour = {"S1": BLUE, "S2": ORANGE, "S3": GREEN}
    shapes.append(text(0.5, 1.55, 6.0, 0.4, [[run("(a)", 14, True), run("  Stored runs: networks by datasets", 14)]],
                       name="panel a"))
    for m in rows:
        shapes.append(text(0.5, ys[m] - 0.04, x0 - 0.62, pitch, [[run(NET.get(m, "17 WHAR Arena networks"), 11)]],
                           align="r", anchor="ctr", name="row label"))
        for dd in dev + held:
            c = cells.get((m, dd))
            bx, by = xs[dd] + (pitch - size) / 2, ys[m] + (pitch - size) / 2
            if c is None:
                shapes.append(box(bx, by, size, size, fill=None, line=GRID, lw=0.5, name="empty cell"))
            elif c["excluded"]:
                shapes.append(box(bx, by, size, size, line=colour[c["set"]], lw=0.5,
                                  pattern=(colour[c["set"]], tint(colour[c["set"]], 0.25)), name="excluded cell"))
            else:
                shapes.append(box(bx, by, size, size, fill=colour[c["set"]], name=f"{c['set']} cell"))
    x_end = xs[held[-1]] + pitch
    for lo, hi, lab in ((x0, x0 + len(dev) * pitch, "development datasets"), (xs[held[0]], x_end, "held-out datasets")):
        shapes.append(line(lo + 0.03, y0 - 0.1, hi - 0.03, y0 - 0.1, lw=1.0, name="group rule"))
        shapes.append(text(lo, y0 - 0.48, hi - lo, 0.34, [[run(lab, 12)]], align="ctr", anchor="b", name="group label"))
    y_lab = ys["whar17"] + pitch + 0.05
    for dd in dev + held:
        shapes.append(text(xs[dd], y_lab, pitch, 1.45, [[run(names[dd]["name"], 10)]], align="r", anchor="ctr",
                           vert=True, name="column label"))

    # (b) UCI-HAR test groups
    tests = d["uci_har_test_subjects"]
    splits = sorted(tests)
    n_subj = 30
    bx0, cw = 9.25, 0.12
    shapes.append(text(8.75, 1.55, 4.3, 0.4, [[run("(b)", 14, True), run("  Person-disjoint splits, UCI-HAR", 14)]],
                       name="panel b"))
    for k, (fill, lab) in enumerate(((BLUE, "test group"), (LIGHT, "training and validation"))):
        kx = bx0 + 0.45 + k * 1.3
        shapes.append(box(kx, 2.04, 0.18, 0.18, fill=fill, name="legend key"))
        shapes.append(text(kx + 0.24, 1.95, 1.75, 0.36, [[run(lab, 11)]], anchor="ctr", name="legend label"))
    for i, sid in enumerate(splits):
        yy = y0 + i * pitch
        shapes.append(text(8.75, yy - 0.04, 0.42, pitch, [[run(str(i), 10)]], align="r", anchor="ctr",
                           name="split label"))
        for subj in range(n_subj):
            t = subj in tests[sid]
            shapes.append(box(bx0 + subj * cw + 0.01, yy + (pitch - size) / 2, cw - 0.02, size,
                              fill=BLUE if t else LIGHT, name="subject cell"))
    yb = y0 + len(splits) * pitch + 0.04
    for s in range(0, n_subj, 5):
        shapes.append(text(bx0 + s * cw - 0.2, yb, cw + 0.4, 0.3, [[run(str(s), 10)]], align="ctr", name="tick"))
    shapes.append(text(bx0, yb + 0.3, n_subj * cw, 0.32, [[run("UCI-HAR subject (of 30)", 11)]], align="ctr",
                       name="axis label"))
    shapes.append(text(8.75 - 0.05, y0 - 0.48, 0.5, 0.34, [[run("split", 11)]], align="r", anchor="b",
                       name="axis label"))
    shapes.append(text(bx0, yb + 0.72, n_subj * cw, 0.6,
                       [[run("Each split tests three subjects never seen in its training or validation data; every "
                             "subject is tested once.", 10, colour=INK2)]], name="note b"))
    # footnote
    shapes.append(text(0.5, 6.92, 12.3, 0.42,
                       [[run("S1 serves the exploratory and sensitivity analyses, S2 the pre-registered test H1, S3 the "
                             "check of how far the result extends. A filled cell holds every split of the dataset "
                             "(three seeds in S1 and S2, one run in S3).", 10, colour=INK2)]], name="footnote"))
    return shapes


# ------------------------------------------------------------------------------------------------------ slide 2
def slide_skeleton() -> list[dict]:
    shapes = [text(0.5, 0.3, 12.3, 0.6, [[run("The shared skeleton of the ten networks", 24, True)]], name="title")]
    y, h, gap = 2.2, 1.05, 0.6
    spec = [("Window", "C × L", 1.25, None), ("Encoder", "linear, C → D", 1.5, None),
            ("N × residual block", "the network's memory", 2.2, ORANGE), ("Mean pooling", "over time", 1.45, None),
            ("Concat + norm", "LayerNorm", 1.55, None), ("Linear head", "K classes", 1.38, None)]
    xs, x = [], 0.5
    for title, sub, w, accent in spec:
        ec = accent or INK2
        shapes.append(labelled(box(x, y, w, h, fill=tint(accent, 0.14) if accent else WHITE, line=ec, lw=1.5,
                                   round_=True, name=title),
                               [[run(title, 13, True)], [run(sub, 12, italic=sub[0] in "CK", colour=INK2)]]))
        xs.append((x, w))
        x += w + gap
    mid = y + h / 2
    for (xa, wa), (xb, _), lab in zip(xs, xs[1:], ["L × C", "L × D", "L × D", "D", "D′"]):
        shapes.append(line(xa + wa, mid, xb, mid, arrow=True, name="edge"))
        shapes.append(text(xa + wa + 0.02, mid - 0.38, gap - 0.04, 0.32, [[run(lab, 11, italic=True)]], align="ctr",
                           anchor="b", name="edge label"))
    (xn, wn), (xh, _) = xs[4], xs[5]
    shapes.append(text(xn + wn - 0.5, y + h + 0.06, gap + 1.0, 0.32, [[run("embedding", 12, True)]], align="ctr",
                       name="embedding label"))
    # side path
    (xw, ww), (xb_, _), (xp, wp) = xs[0], xs[2], xs[3]
    sx, sw, sy, sh = xb_ + 0.2, xp + wp - xb_ - 0.4, 4.2, 0.95
    shapes.append(labelled(box(sx, sy, sw, sh, fill=tint(YELLOW, 0.14), line=YELLOW, lw=1.5, round_=True,
                               name="side path"),
                           [[run("Side path (4 networks)", 13, True)],
                            [run("window statistics or law invariants", 12, colour=INK2)]]))
    cx, ly = xn + wn / 2, sy + sh / 2
    shapes += [line(xw + ww / 2, y + h, xw + ww / 2, ly, name="side line"),
               line(xw + ww / 2, ly, sx, ly, name="side line"),
               line(sx + sw, ly, cx, ly, name="side line"),
               line(cx, ly, cx, y + h, arrow=True, name="side arrow"),
               text(cx + 0.1, ly - 0.75, 0.8, 0.32, [[run("+32", 12, italic=True)]], name="side label")]
    shapes.append(text(0.5, 6.0, 12.3, 0.8,
                       [[run("D′ = D, or D + 32 with a side path. The networks differ only in the memory inside the "
                             "residual blocks and in the optional side path (Table 2); the embedding analysed in the "
                             "letter is the head's input.", 11, colour=INK2)]], name="footnote"))
    return shapes


# ------------------------------------------------------------------------------------------------------ OOXML
NS = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"')
HDR = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
E = lambda inches: int(round(inches * EMU))  # noqa: E731


def _fill(shape) -> str:
    if shape.get("pattern"):
        fg, bg = shape["pattern"]
        return (f'<a:pattFill prst="wdUpDiag"><a:fgClr><a:srgbClr val="{fg}"/></a:fgClr>'
                f'<a:bgClr><a:srgbClr val="{bg}"/></a:bgClr></a:pattFill>')
    return f'<a:solidFill><a:srgbClr val="{shape["fill"]}"/></a:solidFill>' if shape.get("fill") else "<a:noFill/>"


def _ln(colour, lw, arrow=False) -> str:
    if not colour:
        return "<a:ln><a:noFill/></a:ln>"
    tail = '<a:tailEnd type="triangle" w="med" len="med"/>' if arrow else ""
    return f'<a:ln w="{int(lw * 12700)}"><a:solidFill><a:srgbClr val="{colour}"/></a:solidFill>{tail}</a:ln>'


def _txbody(shape) -> str:
    if not shape.get("paras"):
        return ""
    vert = ' vert="vert270"' if shape.get("vert") else ""
    out = [f'<p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" anchor="{shape["anchor"]}"'
           f'{vert} rtlCol="0"><a:noAutofit/></a:bodyPr><a:lstStyle/>']
    for para in shape["paras"]:
        out.append(f'<a:p><a:pPr algn="{shape["align"]}"/>')
        for s, size, bold, italic, colour in para:
            out.append(f'<a:r><a:rPr lang="en-US" sz="{int(size * 100)}" b="{int(bold)}" i="{int(italic)}" dirty="0">'
                       f'<a:solidFill><a:srgbClr val="{colour}"/></a:solidFill><a:latin typeface="{FONT}"/>'
                       f'<a:cs typeface="{FONT}"/></a:rPr><a:t>{escape(s)}</a:t></a:r>')
        out.append(f'<a:endParaRPr lang="en-US" sz="{int(para[-1][1] * 100)}" dirty="0"/></a:p>')
    out.append("</p:txBody>")
    return "".join(out)


def _shape_xml(shape, sid: int) -> str:
    if shape["kind"] == "line":
        x0, y0, x1, y1 = shape["x0"], shape["y0"], shape["x1"], shape["y1"]
        flip = (' flipH="1"' if x1 < x0 else "") + (' flipV="1"' if y1 < y0 else "")
        return (f'<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{sid}" name="{escape(shape["name"])} {sid}"/><p:cNvCxnSpPr/><p:nvPr/>'
                f'</p:nvCxnSpPr><p:spPr><a:xfrm{flip}><a:off x="{E(min(x0, x1))}" y="{E(min(y0, y1))}"/>'
                f'<a:ext cx="{E(abs(x1 - x0))}" cy="{E(abs(y1 - y0))}"/></a:xfrm><a:prstGeom prst="line"><a:avLst/>'
                f'</a:prstGeom>{_ln(shape["colour"], shape["lw"], shape["arrow"])}</p:spPr></p:cxnSp>')
    is_text = shape["kind"] == "text"
    geom = ('<a:prstGeom prst="roundRect"><a:avLst><a:gd name="adj" fmla="val 12000"/></a:avLst></a:prstGeom>'
            if shape.get("round") else '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>')
    fill = "<a:noFill/>" if is_text else _fill(shape)
    ln = "<a:ln><a:noFill/></a:ln>" if is_text else _ln(shape.get("line"), shape.get("lw", 1.0))
    txbox = ' txBox="1"' if is_text else ""
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="{escape(shape["name"])} {sid}"/>'
            f'<p:cNvSpPr{txbox}/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm>'
            f'<a:off x="{E(shape["x"])}" y="{E(shape["y"])}"/><a:ext cx="{E(shape["w"])}" cy="{E(shape["h"])}"/>'
            f'</a:xfrm>{geom}{fill}{ln}</p:spPr>{_txbody(shape)}</p:sp>')


def _slide_xml(shapes) -> str:
    body = "".join(_shape_xml(s, k + 2) for k, s in enumerate(shapes))
    return (f'{HDR}<p:sld {NS}><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/>'
            f'</p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/>'
            f'<a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>{body}</p:spTree></p:cSld><p:clrMapOvr>'
            f'<a:masterClrMapping/></p:clrMapOvr></p:sld>')


def _rels(items) -> str:
    rel = "".join(f'<Relationship Id="{i}" Type="{t}" Target="{g}"/>' for i, t, g in items)
    return f'{HDR}<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rel}</Relationships>'


def _theme() -> str:
    clr = [("dk1", INK), ("lt1", WHITE), ("dk2", INK2), ("lt2", LIGHT), ("accent1", BLUE), ("accent2", ORANGE),
           ("accent3", GREEN), ("accent4", YELLOW), ("accent5", "D55181"), ("accent6", "4A3AA7"),
           ("hlink", BLUE), ("folHlink", "4A3AA7")]
    scheme = "".join(f'<a:{k}><a:srgbClr val="{v}"/></a:{k}>' for k, v in clr)
    font = f'<a:latin typeface="{FONT}"/><a:ea typeface=""/><a:cs typeface=""/>'
    solid = '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
    ln_style = f'<a:ln w="6350">{solid}</a:ln>'
    return (f'{HDR}<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="HARFactors">'
            f'<a:themeElements><a:clrScheme name="HARFactors">{scheme}</a:clrScheme>'
            f'<a:fontScheme name="HARFactors"><a:majorFont>{font}</a:majorFont><a:minorFont>{font}</a:minorFont>'
            f'</a:fontScheme><a:fmtScheme name="HARFactors"><a:fillStyleLst>{solid * 3}</a:fillStyleLst>'
            f'<a:lnStyleLst>{ln_style * 3}'
            f'</a:lnStyleLst><a:effectStyleLst>{"<a:effectStyle><a:effectLst/></a:effectStyle>" * 3}'
            f'</a:effectStyleLst><a:bgFillStyleLst>{solid * 3}</a:bgFillStyleLst></a:fmtScheme></a:themeElements>'
            f'<a:objectDefaults/><a:extraClrSchemeLst/></a:theme>')


def write_pptx(slides: list[list[dict]], path) -> None:
    group = ('<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm>'
             '<a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm>'
             '</p:grpSpPr>')
    clrmap = ('<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" '
              'accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" '
              'folHlink="folHlink"/>')
    master = (f'{HDR}<p:sldMaster {NS}><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="{WHITE}"/></a:solidFill>'
              f'<a:effectLst/></p:bgPr></p:bg><p:spTree>{group}</p:spTree></p:cSld>{clrmap}<p:sldLayoutIdLst>'
              f'<p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst></p:sldMaster>')
    layout = (f'{HDR}<p:sldLayout {NS} type="blank" preserve="1"><p:cSld name="Blank"><p:spTree>{group}'
              f'</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>')
    n = len(slides)
    sld_ids = "".join(f'<p:sldId id="{256 + k}" r:id="rId{2 + k}"/>' for k in range(n))
    pres = (f'{HDR}<p:presentation {NS} saveSubsetFonts="1"><p:sldMasterIdLst><p:sldMasterId id="2147483648" '
            f'r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>{sld_ids}</p:sldIdLst><p:sldSz cx="{E(W_IN)}" '
            f'cy="{E(H_IN)}"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>')
    pres_rels = ([("rId1", f"{REL}/slideMaster", "slideMasters/slideMaster1.xml")]
                 + [(f"rId{2 + k}", f"{REL}/slide", f"slides/slide{k + 1}.xml") for k in range(n)]
                 + [(f"rId{2 + n}", f"{REL}/theme", "theme/theme1.xml"),
                    (f"rId{3 + n}", f"{REL}/presProps", "presProps.xml"),
                    (f"rId{4 + n}", f"{REL}/viewProps", "viewProps.xml"),
                    (f"rId{5 + n}", f"{REL}/tableStyles", "tableStyles.xml")])
    ct = "application/vnd.openxmlformats-officedocument.presentationml"
    overrides = [("/ppt/presentation.xml", f"{ct}.presentation.main+xml"),
                 ("/ppt/slideMasters/slideMaster1.xml", f"{ct}.slideMaster+xml"),
                 ("/ppt/slideLayouts/slideLayout1.xml", f"{ct}.slideLayout+xml"),
                 ("/ppt/theme/theme1.xml", "application/vnd.openxmlformats-officedocument.theme+xml"),
                 ("/ppt/presProps.xml", f"{ct}.presProps+xml"), ("/ppt/viewProps.xml", f"{ct}.viewProps+xml"),
                 ("/ppt/tableStyles.xml", f"{ct}.tableStyles+xml"),
                 ("/docProps/core.xml", "application/vnd.openxmlformats-package.core-properties+xml"),
                 ("/docProps/app.xml", "application/vnd.openxmlformats-officedocument.extended-properties+xml")]
    overrides += [(f"/ppt/slides/slide{k + 1}.xml", f"{ct}.slide+xml") for k in range(n)]
    types = (f'{HDR}<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
             '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
             '<Default Extension="xml" ContentType="application/xml"/>'
             + "".join(f'<Override PartName="{p}" ContentType="{c}"/>' for p, c in overrides) + "</Types>")
    core = (f'{HDR}<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>HARFactors: Figures 2 and 3</dc:title>'
            '<dc:creator>HARFactors (src/visualization/pptx_export.py)</dc:creator>'
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}'
            '</dcterms:created></cp:coreProperties>')
    app = (f'{HDR}<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
           f'<Application>HARFactors</Application><Slides>{n}</Slides></Properties>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", types)
        z.writestr("_rels/.rels", _rels([("rId1", f"{REL}/officeDocument", "ppt/presentation.xml"),
                                         ("rId2", "http://schemas.openxmlformats.org/package/2006/relationships/"
                                                  "metadata/core-properties", "docProps/core.xml"),
                                         ("rId3", f"{REL}/extended-properties", "docProps/app.xml")]))
        z.writestr("docProps/core.xml", core)
        z.writestr("docProps/app.xml", app)
        z.writestr("ppt/presentation.xml", pres)
        z.writestr("ppt/_rels/presentation.xml.rels", _rels(pres_rels))
        z.writestr("ppt/presProps.xml", f"{HDR}<p:presentationPr {NS}/>")
        z.writestr("ppt/viewProps.xml", f"{HDR}<p:viewPr {NS}/>")
        z.writestr("ppt/tableStyles.xml", f'{HDR}<a:tblStyleLst xmlns:a="http://schemas.openxmlformats.org/drawingml/'
                                          f'2006/main" def="{{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}}"/>')
        z.writestr("ppt/theme/theme1.xml", _theme())
        z.writestr("ppt/slideMasters/slideMaster1.xml", master)
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels",
                   _rels([("rId1", f"{REL}/slideLayout", "../slideLayouts/slideLayout1.xml"),
                          ("rId2", f"{REL}/theme", "../theme/theme1.xml")]))
        z.writestr("ppt/slideLayouts/slideLayout1.xml", layout)
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels",
                   _rels([("rId1", f"{REL}/slideMaster", "../slideMasters/slideMaster1.xml")]))
        for k, shapes in enumerate(slides):
            z.writestr(f"ppt/slides/slide{k + 1}.xml", _slide_xml(shapes))
            z.writestr(f"ppt/slides/_rels/slide{k + 1}.xml.rels",
                       _rels([("rId1", f"{REL}/slideLayout", "../slideLayouts/slideLayout1.xml")]))


# ------------------------------------------------------------------------------------------------------ preview
def preview(shapes, path) -> None:
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=110)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_IN)
    ax.set_ylim(H_IN, 0)
    ax.axis("off")
    hx = lambda c: "#" + c  # noqa: E731
    for s in shapes:
        if s["kind"] == "line":
            ax.annotate("", xy=(s["x1"], s["y1"]), xytext=(s["x0"], s["y0"]),
                        arrowprops={"arrowstyle": "-|>" if s["arrow"] else "-", "color": hx(s["colour"]),
                                    "lw": s["lw"], "shrinkA": 0, "shrinkB": 0, "mutation_scale": 12})
            continue
        if s["kind"] == "box":
            kw = {"fc": hx(s["fill"]) if s.get("fill") else "none", "ec": hx(s["line"]) if s.get("line") else "none",
                  "lw": s.get("lw", 1.0)}
            if s.get("pattern"):
                kw |= {"fc": hx(s["pattern"][1]), "hatch": "////", "ec": hx(s["pattern"][0])}
            patch = (FancyBboxPatch((s["x"], s["y"]), s["w"], s["h"], boxstyle="round,pad=0,rounding_size=0.12", **kw)
                     if s.get("round") else Rectangle((s["x"], s["y"]), s["w"], s["h"], **kw))
            ax.add_patch(patch)
        if s.get("paras"):
            n = len(s["paras"])
            sizes = [max(r[1] for r in p) for p in s["paras"]]
            total = sum(sz / 72 * 1.2 for sz in sizes)
            if s.get("vert"):
                xx = s["x"] + s["w"] / 2
                yy = {"r": s["y"], "l": s["y"] + s["h"], "ctr": s["y"] + s["h"] / 2}[s["align"]]
                ax.text(xx, yy, "".join(r[0] for r in s["paras"][0]), rotation=90, fontsize=sizes[0],
                        va={"r": "top", "l": "bottom", "ctr": "center"}[s["align"]], ha="center", color=hx(INK))
                continue
            top = {"t": s["y"], "ctr": s["y"] + (s["h"] - total) / 2, "b": s["y"] + s["h"] - total}[s["anchor"]]
            xx = {"l": s["x"], "ctr": s["x"] + s["w"] / 2, "r": s["x"] + s["w"]}[s["align"]]
            ha = {"l": "left", "ctr": "center", "r": "right"}[s["align"]]
            yy = top
            for p, sz in zip(s["paras"], sizes):
                txt = "".join(r[0] for r in p)
                ax.text(xx, yy, txt, fontsize=sz, ha=ha, va="top", color=hx(p[-1][4]),
                        fontweight="bold" if all(r[2] for r in p) else "normal", wrap=True)
                yy += sz / 72 * 1.2
            if n and s["kind"] == "text":
                ax.add_patch(Rectangle((s["x"], s["y"]), s["w"], s["h"], fc="none", ec="#ff00ff", lw=0.3, ls=":"))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    slides = [slide_design(), slide_skeleton()]
    PATHS.FIGURES.mkdir(exist_ok=True)
    write_pptx(slides, OUT)
    for k, shapes in enumerate(slides):
        preview(shapes, PATHS.FIGURES / f"fig02_fig03_slide{k + 1}.png")
    print("wrote", OUT, "with", [len(s) for s in slides], "shapes")


if __name__ == "__main__":
    main()
