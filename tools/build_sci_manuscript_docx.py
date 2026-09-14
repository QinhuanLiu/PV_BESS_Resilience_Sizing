from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MD = ROOT / "manuscript_core_body_no_intro.md"
OUT_DOCX = ROOT / "PV_BESS_resilience_manuscript_core_with_figures.docx"

FIGURES = {
    2: ROOT / "outputs/SG/SG_UTown_Resilience_Frontier/SG_UTown_Figure_2_Input_Profiles.png",
    3: ROOT / "outputs/SG/SG_UTown_Resilience_Frontier/SG_UTown_Figure_3_Outage_Scenarios.png",
    4: ROOT / "outputs/SG/SG_UTown_Resilience_Frontier/SG_UTown_Figure_4_PV_Building_Load_Mismatch.png",
    5: ROOT / "outputs/SG/SG_UTown_Resilience_Frontier/SG_UTown_Figure_5_Battery_vs_Outage_Load_Served_Ratio.png",
    6: ROOT / "outputs/SG/SG_UTown_Resilience_Frontier/SG_UTown_Figure_6_Battery_vs_UE_Unserved_Cost.png",
    7: ROOT / "outputs/SG/SG_UTown_Resilience_Frontier/SG_UTown_Figure_7_Battery_vs_System_Total_Cost.png",
    8: ROOT / "outputs/SG/SG_UTown_Resilience_Frontier/SG_UTown_Figure_8_Battery_vs_Achieved_LOLP.png",
    9: ROOT / "outputs/SG/SG_UTown_Resilience_Frontier/SG_UTown_Figure_9_Battery_vs_Chance_Constraint_Probability.png",
    10: ROOT / "outputs/SG/SG_UTown_Resilience_Frontier/SG_UTown_Figure_10_Marginal_Cost_Reliability.png",
}

CAPTIONS = {
    2: "Figure 2. Input profiles for the SG_UTown case. The panels summarize representative-week full-building electric-equivalent load, scaled PV generation and the 24 h time-of-use tariff used in the dispatch model.",
    3: "Figure 3. Stochastic outage scenario generation. The figure shows an example grid-availability sequence and the distributions of outage start times and durations across the sampled scenarios.",
    4: "Figure 4. PV-building load mismatch. Empirical distributions of full-building electric-equivalent load and PV generation under selected PV capacities show why storage is required for outage service.",
    5: "Figure 5. Battery capacity versus outage load served ratio. Larger battery capacity increases the mean fraction of outage-weighted load served, with diminishing incremental gains at higher capacities.",
    6: "Figure 6. Battery capacity versus unserved energy and unserved-energy cost. The paired panels link the physical resilience metric to its monetary outage-loss valuation.",
    7: "Figure 7. Battery capacity versus annual system total cost. The frontier compares total cost and cost components and marks the distinction between the minimum-cost and reliability-constrained designs.",
    8: "Figure 8. Battery capacity versus achieved LOLP. The horizontal threshold indicates the 0.10 achieved-LOLP target used to identify reliability-constrained candidates.",
    9: "Figure 9. Battery capacity versus chance-constraint probability. The metric reports the proportion of stochastic outage scenarios satisfying the scenario-level LOLP threshold.",
    10: "Figure 10. Marginal cost of reliability improvement. The panels show how incremental cost changes with reductions in achieved LOLP and increases in chance-constraint probability.",
}


def set_font(run, name: str, size: float | None = None, bold: bool | None = None, italic: bool | None = None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:ascii"), name)
    run._element.rPr.rFonts.set(qn("w:hAnsi"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def style_document(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Cm(21.59)
    section.page_height = Cm(27.94)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    section.header_distance = Inches(0.5)
    section.footer_distance = Inches(0.5)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.first_line_indent = Cm(0.5)

    for name, size, color, before, after in [
        ("Heading 1", 14, "000000", 12, 6),
        ("Heading 2", 12, "000000", 10, 4),
        ("Heading 3", 12, "000000", 8, 3),
    ]:
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.line_spacing = 1.5
        style.paragraph_format.keep_with_next = True

    caption = doc.styles.add_style("Figure Caption", 1)
    caption.font.name = "Times New Roman"
    caption._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    caption._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    caption.font.size = Pt(10)
    caption.paragraph_format.line_spacing = 1.0
    caption.paragraph_format.space_before = Pt(3)
    caption.paragraph_format.space_after = Pt(10)
    caption.paragraph_format.first_line_indent = Pt(0)

    equation = doc.styles.add_style("Display Equation", 1)
    equation.font.name = "Times New Roman"
    equation._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    equation._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    equation.font.size = Pt(12)
    equation.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    equation.paragraph_format.first_line_indent = Pt(0)
    equation.paragraph_format.space_before = Pt(3)
    equation.paragraph_format.space_after = Pt(8)
    equation.paragraph_format.line_spacing = 1.0


def add_field(paragraph, field: str) -> None:
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = field
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)


def add_header_footer(doc: Document) -> None:
    section = doc.sections[0]
    header_p = section.header.paragraphs[0]
    header_p.text = "PV-BESS resilience manuscript"
    header_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header_p.paragraph_format.space_after = Pt(0)
    for run in header_p.runs:
        set_font(run, "Times New Roman", 9)
        run.font.color.rgb = RGBColor(90, 90, 90)

    footer_p = section.footer.paragraphs[0]
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer_p.paragraph_format.space_after = Pt(0)
    footer_p.add_run("Page ")
    add_field(footer_p, "PAGE")
    for run in footer_p.runs:
        set_font(run, "Times New Roman", 9)
        run.font.color.rgb = RGBColor(90, 90, 90)


def add_title_block(doc: Document) -> None:
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.first_line_indent = Pt(0)
    title.paragraph_format.space_after = Pt(8)
    run = title.add_run("PV-BESS Resilience Cost-Reliability Assessment for Building-Community Outage Resilience")
    set_font(run, "Times New Roman", 16, bold=True)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.first_line_indent = Pt(0)
    subtitle.paragraph_format.space_after = Pt(12)
    run = subtitle.add_run("Core manuscript body excluding Introduction and Literature Review")
    set_font(run, "Times New Roman", 11, italic=True)

    note = doc.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    note.paragraph_format.first_line_indent = Pt(0)
    note.paragraph_format.space_after = Pt(14)
    run = note.add_run("Prepared from SG_UTown model outputs with embedded Figures 2-10 and 13-15.")
    set_font(run, "Times New Roman", 10)


def add_text_paragraph(doc: Document, text: str) -> None:
    p = doc.add_paragraph(style="Normal")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Cm(0.5)
    text = clean_inline_text(text)
    for part in split_inline_code(text):
        run = p.add_run(part[0])
        set_font(run, "Times New Roman", 12)
        if part[1]:
            run.italic = True


def clean_inline_text(text: str) -> str:
    text = re.sub(r"\\\((.+?)\\\)", r"\1", text)
    text = text.replace("\\leq", "<=")
    text = text.replace("\\geq", ">=")
    return text


def split_inline_code(text: str):
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return [(text, False)]


def add_heading(doc: Document, text: str, level: int) -> None:
    if text == "Core Manuscript Text Excluding Introduction":
        return
    p = doc.add_paragraph(text, style=f"Heading {level}")
    p.paragraph_format.first_line_indent = Pt(0)


def add_equation(doc: Document, raw: str) -> None:
    text = raw.strip()
    if "L_t" in text:
        text = "L_t^eq = L_t^electric + L_t^cooling / COP_c + L_t^heating / COP_h"
    text = text.replace("\\", "").replace("{", "").replace("}", "")
    p = doc.add_paragraph(style="Display Equation")
    p.add_run(text)
    for run in p.runs:
        set_font(run, "Times New Roman", 12, italic=True)


def add_figure(doc: Document, fig_no: int) -> None:
    path = FIGURES[fig_no]
    if not path.exists():
        return
    width = Inches(6.1)
    if fig_no in {2, 3}:
        width = Inches(5.65)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(str(path), width=width)

    cap = doc.add_paragraph(style="Figure Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    cap.paragraph_format.first_line_indent = Pt(0)
    bold_prefix = f"Figure {fig_no}."
    text = CAPTIONS[fig_no]
    if text.startswith(bold_prefix):
        r1 = cap.add_run(bold_prefix)
        set_font(r1, "Times New Roman", 10, bold=True)
        r2 = cap.add_run(text[len(bold_prefix):])
        set_font(r2, "Times New Roman", 10)
    else:
        r = cap.add_run(text)
        set_font(r, "Times New Roman", 10)


def trigger_figures(text: str) -> list[int]:
    mapping = [
        ("Figure 2", 2),
        ("Figure 3", 3),
        ("Figure 4", 4),
        ("Figure 5", 5),
        ("Figure 6", 6),
        ("Figure 7", 7),
        ("Figure 8", 8),
        ("Figure 9", 9),
        ("Figure 10", 10),
    ]
    return [fig for key, fig in mapping if key in text]


def add_key_results_table(doc: Document) -> None:
    h = doc.add_paragraph("Key numerical results used in the manuscript", style="Heading 2")
    h.paragraph_format.first_line_indent = Pt(0)
    table = doc.add_table(rows=1, cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    headers = ["Design", "PV capacity", "BESS capacity", "Main performance"]
    for i, value in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = value
        shade_cell(cell, "F2F2F2")
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for p in cell.paragraphs:
            p.paragraph_format.first_line_indent = Pt(0)
            for r in p.runs:
                set_font(r, "Times New Roman", 10, bold=True)
    rows = [
        ["Minimum-total-cost design", "20 MW", "10 MWh", "System total cost 8.69 million SGD yr-1; achieved LOLP 0.193; chance-constraint probability 0.58."],
        ["Reliability-constrained design", "28 MW", "20 MWh", "System total cost 9.01 million SGD yr-1; achieved LOLP 0.097; chance-constraint probability 0.75."],
    ]
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = value
            cells[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cells[i].paragraphs:
                p.paragraph_format.first_line_indent = Pt(0)
                p.paragraph_format.space_after = Pt(2)
                for r in p.runs:
                    set_font(r, "Times New Roman", 10)
    widths = [1.55, 1.0, 1.55, 2.1]
    for row in table.rows:
        for i, width in enumerate(widths):
            row.cells[i].width = Inches(width)
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(0)


def shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def build() -> None:
    doc = Document()
    style_document(doc)
    add_header_footer(doc)
    add_title_block(doc)

    text = SOURCE_MD.read_text(encoding="utf-8")
    text = text.split("## Claim-evidence map")[0].strip()
    lines = text.splitlines()
    paragraph_buffer: list[str] = []
    equation_buffer: list[str] = []
    in_equation = False

    def flush_paragraph():
        if not paragraph_buffer:
            return
        paragraph = " ".join(line.strip() for line in paragraph_buffer).strip()
        paragraph_buffer.clear()
        if paragraph:
            add_text_paragraph(doc, paragraph)
            for fig in trigger_figures(paragraph):
                add_figure(doc, fig)

    for line in lines:
        stripped = line.strip()
        if stripped == r"\[":
            flush_paragraph()
            in_equation = True
            equation_buffer = []
            continue
        if stripped == r"\]":
            add_equation(doc, " ".join(equation_buffer))
            in_equation = False
            equation_buffer = []
            continue
        if in_equation:
            equation_buffer.append(stripped)
            continue
        if not stripped:
            flush_paragraph()
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            add_heading(doc, stripped[4:], 3)
        elif stripped.startswith("## "):
            flush_paragraph()
            add_heading(doc, stripped[3:], 1)
        elif stripped.startswith("# "):
            flush_paragraph()
            add_heading(doc, stripped[2:], 1)
        else:
            paragraph_buffer.append(stripped)
    flush_paragraph()

    doc.add_section(WD_SECTION.NEW_PAGE)
    add_key_results_table(doc)
    doc.save(OUT_DOCX)


if __name__ == "__main__":
    build()
