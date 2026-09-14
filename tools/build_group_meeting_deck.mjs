import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = "/Users/qinhuan/Documents/PV_BESS_Resilience_Sizing";
const OUT_DIR = path.join(ROOT, "outputs", "group_meeting_slides");
const PREVIEW_DIR = path.join(OUT_DIR, "preview");
const FINAL_PPTX = path.join(OUT_DIR, "SG_UTown_PV_BESS_resilience_group_meeting.pptx");
const QA_REPORT = path.join(OUT_DIR, "qa_report.txt");

const W = 1280;
const H = 720;
const M = 54;
const INK = "#111111";
const MUTED = "#5B6470";
const RULE = "#CBD5E1";
const PANEL = "#F3F5F7";
const BLUE = "#0F4D92";
const TEAL = "#0F766E";
const ORANGE = "#C45A2A";
const RED = "#B64342";

const figs = {
  2: path.join(ROOT, "outputs/SG/SG_UTown_Resilience_Frontier/SG_UTown_Figure_2_Input_Profiles.png"),
  3: path.join(ROOT, "outputs/SG/SG_UTown_Resilience_Frontier/SG_UTown_Figure_3_Outage_Scenarios.png"),
  4: path.join(ROOT, "outputs/SG/SG_UTown_Resilience_Frontier/SG_UTown_Figure_4_PV_Building_Load_Mismatch.png"),
  5: path.join(ROOT, "outputs/SG/SG_UTown_Resilience_Frontier/SG_UTown_Figure_5_Battery_vs_Outage_Load_Served_Ratio.png"),
  6: path.join(ROOT, "outputs/SG/SG_UTown_Resilience_Frontier/SG_UTown_Figure_6_Battery_vs_UE_Unserved_Cost.png"),
  7: path.join(ROOT, "outputs/SG/SG_UTown_Resilience_Frontier/SG_UTown_Figure_7_Battery_vs_System_Total_Cost.png"),
  8: path.join(ROOT, "outputs/SG/SG_UTown_Resilience_Frontier/SG_UTown_Figure_8_Battery_vs_Achieved_LOLP.png"),
  9: path.join(ROOT, "outputs/SG/SG_UTown_Resilience_Frontier/SG_UTown_Figure_9_Battery_vs_Chance_Constraint_Probability.png"),
  10: path.join(ROOT, "outputs/SG/SG_UTown_Resilience_Frontier/SG_UTown_Figure_10_Marginal_Cost_Reliability.png"),
};

async function bytes(file) {
  return await fs.readFile(file);
}

async function writeBlob(file, blob) {
  await fs.writeFile(file, new Uint8Array(await blob.arrayBuffer()));
}

function addBox(slide, { left, top, width, height, fill = PANEL, line = "none", radius = 0 }) {
  return slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height },
    fill,
    line: line === "none" ? { style: "solid", fill: "none", width: 0 } : line,
    borderRadius: radius,
  });
}

function addText(slide, text, pos, style = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position: pos,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    fontSize: style.fontSize ?? 22,
    bold: style.bold ?? false,
    color: style.color ?? INK,
    alignment: style.alignment ?? "left",
  };
  return box;
}

function addTitle(slide, title, eyebrow = "") {
  if (eyebrow) {
    addText(slide, eyebrow, { left: M, top: 28, width: 460, height: 24 }, { fontSize: 16, bold: true, color: MUTED });
  }
  addText(slide, title, { left: M, top: 58, width: 980, height: 88 }, { fontSize: 38, bold: true, color: INK });
  slide.shapes.add({
    geometry: "line",
    position: { left: M, top: 145, width: 1172, height: 0 },
    fill: "none",
    line: { style: "solid", fill: RULE, width: 1 },
  });
}

function addFooter(slide, page, total, source = "Source: SG_UTown model outputs") {
  addText(slide, source, { left: M, top: 674, width: 760, height: 24 }, { fontSize: 14, color: MUTED });
  addText(slide, `${page}/${total}`, { left: 1160, top: 674, width: 70, height: 24 }, { fontSize: 14, color: MUTED, alignment: "right" });
}

async function addFigure(slide, figNo, pos, options = {}) {
  addBox(slide, { left: pos.left - 8, top: pos.top - 8, width: pos.width + 16, height: pos.height + 16, fill: "#FFFFFF", line: { style: "solid", fill: "#D6DEE8", width: 1 } });
  slide.images.add({
    blob: await bytes(figs[figNo]),
    contentType: "image/png",
    alt: `Figure ${figNo}`,
    fit: options.fit ?? "contain",
    position: pos,
  });
}

function addBullets(slide, items, pos, color = INK) {
  const text = items.map((item) => `• ${item}`).join("\n");
  return addText(slide, text, pos, { fontSize: 22, color });
}

function addMetric(slide, value, label, pos, color = BLUE) {
  addText(slide, value, { left: pos.left, top: pos.top, width: pos.width, height: 54 }, { fontSize: 42, bold: true, color });
  addText(slide, label, { left: pos.left, top: pos.top + 58, width: pos.width, height: 54 }, { fontSize: 18, color: MUTED });
}

function addNote(slide, note) {
  slide.speakerNotes.textFrame.setText(note);
  slide.speakerNotes.setVisible(true);
}

function addTakeaway(slide, text, color = "#F8FAFC") {
  addBox(slide, { left: M, top: 608, width: 1172, height: 52, fill: color, line: { style: "solid", fill: "#DDE4EC", width: 1 } });
  addText(slide, text, { left: M + 18, top: 620, width: 1136, height: 28 }, { fontSize: 20, bold: true, color: INK });
}

async function build() {
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  const pres = Presentation.create({ slideSize: { width: W, height: H } });
  const slides = [];
  const add = () => {
    const s = pres.slides.add();
    s.background.fill = "#FFFFFF";
    slides.push(s);
    return s;
  };

  let s = add();
  addText(s, "PV-BESS 韧性容量配置：成本-可靠性前沿", { left: 70, top: 118, width: 1060, height: 150 }, { fontSize: 48, bold: true });
  addText(s, "SG_UTown building-community case | 组会汇报", { left: 72, top: 294, width: 760, height: 34 }, { fontSize: 24, color: MUTED });
  addBox(s, { left: 72, top: 392, width: 1140, height: 112, fill: "#F7F9FB", line: { style: "solid", fill: "#DDE4EC", width: 1 } });
  addText(s, "核心问题", { left: 98, top: 416, width: 160, height: 32 }, { fontSize: 24, bold: true, color: BLUE });
  addText(s, "How much PV and battery capacity is economically and reliably justified under stochastic outages?", { left: 260, top: 414, width: 890, height: 56 }, { fontSize: 24, color: INK });
  addText(s, "Based on manuscript core body and Figures 2–10", { left: 72, top: 618, width: 780, height: 28 }, { fontSize: 18, color: MUTED });
  addFooter(s, 1, 13, "Source: manuscript_core_body_no_intro.md; SG_UTown outputs");
  addNote(s, "开场说明：这里聚焦单一 outage-resilience sizing 口径，把 outage resilience 作为成本和可靠性共同约束的问题。");

  s = add();
  addTitle(s, "为什么不能只看最低成本 sizing？", "研究动机");
  addText(s, "PV-BESS 的同一块电池同时面对两个目标：", { left: 74, top: 190, width: 670, height: 36 }, { fontSize: 30, bold: true });
  addBullets(s, ["正常运行：削减 TOU 电费", "停电期间：支撑 building load", "两者共享同一 SOC 空间"], { left: 90, top: 252, width: 520, height: 150 });
  addBox(s, { left: 760, top: 190, width: 360, height: 300, fill: "#F3F7FA", line: { style: "solid", fill: "#DDE4EC", width: 1 } });
  addMetric(s, "cost ≠ reliability", "最低年化成本设计不一定满足 LOLP 目标", { left: 800, top: 240, width: 280, height: 110 }, RED);
  addMetric(s, "cost ≠ target", "最低年化成本设计可能不能满足可靠性门槛", { left: 800, top: 376, width: 280, height: 110 }, TEAL);
  addTakeaway(s, "组会关注点：用 frontier 同时回答成本、UE、LOLP 和场景一致性。");
  addFooter(s, 2, 13);
  addNote(s, "这一页用于建立听众心智模型：成本最优和可靠性约束是两个不同决策。");

  s = add();
  addTitle(s, "Frontier-based framework 聚焦容量配置", "方法框架");
  const stepY = 224;
  const stepW = 300;
  const xs = [118, 490, 862];
  const labels = [
    ["Input", "8760 h load / PV / TOU\nSAIFI / SAIDI / VoLL\nPV-BESS cost"],
    ["Frontier", "PV × BESS sweep\n100 outage scenarios\ncost-reliability frontier"],
    ["Decision", "minimum-cost design\nreliability-constrained design\nLOLP target = 0.10"],
  ];
  labels.forEach(([head, body], i) => {
    addBox(s, { left: xs[i], top: stepY, width: stepW, height: 190, fill: i === 1 ? "#EDF7F5" : "#F7F9FB", line: { style: "solid", fill: "#D5DEE8", width: 1 } });
    addText(s, head, { left: xs[i] + 22, top: stepY + 26, width: stepW - 44, height: 38 }, { fontSize: 26, bold: true, color: i === 1 ? TEAL : BLUE });
    addText(s, body, { left: xs[i] + 22, top: stepY + 78, width: stepW - 44, height: 92 }, { fontSize: 20, color: INK });
    if (i < labels.length - 1) {
      s.shapes.add({ geometry: "rightArrow", position: { left: xs[i] + stepW + 10, top: stepY + 76, width: 42, height: 36 }, fill: "#D9E2EC", line: { style: "solid", fill: "none", width: 0 } });
    }
  });
  addTakeaway(s, "输出不是单点答案，而是“成本、UE、LOLP、chance probability”的联合证据链。");
  addFooter(s, 3, 13);
  addNote(s, "解释主流程：输入驱动容量扫描，然后用 frontier 选择 minimum-cost 和 reliability-constrained design。");

  s = add();
  addTitle(s, "SG_UTown 输入显示：负荷、PV 和电价并不同步", "Case input");
  await addFigure(s, 2, { left: 80, top: 176, width: 720, height: 420 });
  addBullets(s, ["8760 h building load", "PV profile scaled by kW", "TOU tariff repeated yearly"], { left: 850, top: 210, width: 310, height: 170 }, INK);
  addTakeaway(s, "韧性问题必须按小时模拟，不能用 annual energy balance 近似。");
  addFooter(s, 4, 13, "Source: Figure 2, SG_UTown input profiles");
  addNote(s, "强调输入驱动：Load/PV/TOU/Parameter 都来自 Input 文件夹，不是合成负荷。");

  s = add();
  addTitle(s, "随机停电影响 PV-BESS 是否真正可用", "Outage scenarios");
  await addFigure(s, 3, { left: 86, top: 174, width: 665, height: 424 });
  addText(s, "关键点", { left: 822, top: 190, width: 260, height: 34 }, { fontSize: 28, bold: true, color: BLUE });
  addBullets(s, ["100 个 stochastic scenarios", "事件开始时间和持续时间不同", "同一容量在不同 outage timing 下表现不同"], { left: 822, top: 246, width: 340, height: 166 });
  addTakeaway(s, "停电时间决定 PV 与 SOC 是否能刚好支撑负荷。");
  addFooter(s, 5, 13, "Source: Figure 3, stochastic outage scenarios");
  addNote(s, "这里要讲为什么 chance-constraint probability 有意义：不是只看平均 outage duration。");

  s = add();
  addTitle(s, "PV alone 不能保证 outage service", "PV-load mismatch");
  await addFigure(s, 4, { left: 78, top: 190, width: 770, height: 360 });
  addMetric(s, "mismatch", "PV 输出与 building load 的分布和时段不同", { left: 900, top: 220, width: 270, height: 120 }, TEAL);
  addText(s, "因此，battery capacity 是把 PV 能量转移到停电时段的关键资源。", { left: 900, top: 376, width: 275, height: 92 }, { fontSize: 23, color: INK });
  addTakeaway(s, "容量问题的重点从“装多少 PV”转向“需要多少 storage 才能可靠服务”。");
  addFooter(s, 6, 13, "Source: Figure 4, PV-building load mismatch");
  addNote(s, "Figure 4 是后面结果的铺垫：PV 与 load mismatch 是需要 BESS 的直接原因。");

  s = add();
  addTitle(s, "增加 BESS 容量能提高 outage load served ratio", "Stage 1 result");
  await addFigure(s, 5, { left: 82, top: 180, width: 770, height: 380 });
  addText(s, "读图方式", { left: 900, top: 190, width: 260, height: 34 }, { fontSize: 28, bold: true, color: BLUE });
  addBullets(s, ["小容量区：改善有限", "有效区：served ratio 快速上升", "高容量区：边际收益递减"], { left: 900, top: 248, width: 310, height: 150 });
  addTakeaway(s, "BESS 并非越大越“线性变好”，存在明显 diminishing returns。");
  addFooter(s, 7, 13, "Source: Figure 5, battery capacity vs outage load served ratio");
  addNote(s, "讲曲线形状，不需要逐条曲线读数；重点是非线性关系。");

  s = add();
  addTitle(s, "UE 把技术韧性转化为经济损失", "Stage 1 result");
  await addFigure(s, 6, { left: 80, top: 178, width: 790, height: 374 });
  addBox(s, { left: 910, top: 210, width: 250, height: 200, fill: "#F8FAFC", line: { style: "solid", fill: "#DDE4EC", width: 1 } });
  addText(s, "UE × VoLL", { left: 938, top: 248, width: 200, height: 42 }, { fontSize: 34, bold: true, color: RED });
  addText(s, "unserved energy cost", { left: 938, top: 306, width: 190, height: 32 }, { fontSize: 20, color: MUTED });
  addText(s, "这使 outage loss 可以和 PV-BESS annual cost 放在同一目标函数下比较。", { left: 920, top: 438, width: 250, height: 92 }, { fontSize: 20, color: INK });
  addTakeaway(s, "韧性价值不是抽象指标，而是 avoided outage loss。");
  addFooter(s, 8, 13, "Source: Figure 6, UE and unserved-energy cost");
  addNote(s, "强调本文采用 VoLL = 50 SGD/kWh；不展开讨论 VoLL 选择，只说明模型如何用它估值。");

  s = add();
  addTitle(s, "最低成本设计不等于可靠性约束设计", "Cost-reliability frontier");
  await addFigure(s, 7, { left: 64, top: 172, width: 690, height: 430 });
  addText(s, "20 MW +\n10 MWh", { left: 800, top: 192, width: 340, height: 92 }, { fontSize: 38, bold: true, color: BLUE });
  addText(s, "minimum-total-cost design\n8.69 million SGD yr⁻¹\nachieved LOLP = 0.193", { left: 802, top: 300, width: 340, height: 82 }, { fontSize: 20, color: MUTED });
  addText(s, "28 MW +\n20 MWh", { left: 800, top: 405, width: 340, height: 92 }, { fontSize: 38, bold: true, color: TEAL });
  addText(s, "reliability-constrained design\n9.01 million SGD yr⁻¹\nachieved LOLP = 0.097", { left: 802, top: 514, width: 340, height: 82 }, { fontSize: 20, color: MUTED });
  addTakeaway(s, "满足 LOLP ≤ 0.10 需要更大的 PV 和 BESS，并付出约 0.32 million SGD yr⁻¹ 的额外成本。");
  addFooter(s, 9, 13, "Source: Figure 7; Frontier_Optimal_Designs.csv");
  addNote(s, "这是整套汇报的核心结果页。重点对比两个 design，而不是解释所有成本线。");

  s = add();
  addTitle(s, "Achieved LOLP 给出平均可靠性门槛", "Reliability metric 1");
  await addFigure(s, 8, { left: 84, top: 182, width: 760, height: 390 });
  addText(s, "LOLP threshold = 0.10", { left: 890, top: 218, width: 300, height: 44 }, { fontSize: 28, bold: true, color: RED });
  addBullets(s, ["最低成本设计：0.193", "可靠性约束设计：0.097", "平均意义上满足目标"], { left: 900, top: 292, width: 310, height: 140 });
  addTakeaway(s, "可靠性约束会把 optimum 从 cost minimum 推向更大容量。");
  addFooter(s, 10, 13, "Source: Figure 8; LOLP threshold = 0.10");
  addNote(s, "强调 achieved LOLP 是 across scenario 的平均值，所以还需要下一页的 chance probability。");

  s = add();
  addTitle(s, "Chance-constraint probability 看的是场景一致性", "Reliability metric 2");
  await addFigure(s, 9, { left: 82, top: 182, width: 740, height: 390 });
  addMetric(s, "58%", "minimum-cost design\n满足 scenario LOLP ≤ 0.10", { left: 885, top: 230, width: 280, height: 122 }, ORANGE);
  addMetric(s, "75%", "reliability-constrained design\n满足 scenario LOLP ≤ 0.10", { left: 885, top: 408, width: 280, height: 122 }, TEAL);
  addTakeaway(s, "平均可靠性改善后，仍有一部分 stochastic outage scenarios 不能满足阈值。");
  addFooter(s, 11, 13, "Source: Figure 9; Frontier_Optimal_Designs.csv");
  addNote(s, "这一页解释为什么本文使用 chance-constraint probability：它告诉我们 reliability target 在多少场景中成立。");

  s = add();
  addTitle(s, "可靠性提升的边际成本并不恒定", "Marginal economics");
  await addFigure(s, 10, { left: 70, top: 176, width: 810, height: 383 });
  addText(s, "决策含义", { left: 920, top: 196, width: 250, height: 34 }, { fontSize: 28, bold: true, color: BLUE });
  addBullets(s, ["有效容量区：avoided outage loss 可抵消部分成本", "过高容量区：可靠性收益变小", "需要选择经济上合理的 reliability target"], { left: 920, top: 254, width: 300, height: 180 });
  addTakeaway(s, "frontier 的价值在于定位“继续加容量是否还值得”。");
  addFooter(s, 12, 13, "Source: Figure 10, marginal reliability cost");
  addNote(s, "讲 marginal structure：不是简单大电池更好，而是额外可靠性成本什么时候变高。");

  s = add();
  addTitle(s, "组会结论：这篇文章的主线已经成立", "Synthesis");
  const synth = [
    ["1", "容量主线", "PV-BESS sizing 应写成 cost–reliability frontier，而不是单点优化。"],
    ["2", "可靠性证据", "UE、achieved LOLP 和 chance probability 必须一起报告。"],
    ["3", "经济结构", "marginal cost 图帮助判断继续增加容量是否还值得。"],
    ["4", "下一步", "补 battery CAPEX / SAIDI–VoLL sensitivity，完善 discussion 和引用。"],
  ];
  synth.forEach(([num, head, body], i) => {
    const y = 176 + i * 95;
    addText(s, num, { left: 88, top: y + 4, width: 42, height: 44 }, { fontSize: 36, bold: true, color: i === 3 ? ORANGE : BLUE });
    addText(s, head, { left: 154, top: y, width: 210, height: 34 }, { fontSize: 26, bold: true, color: INK });
    addText(s, body, { left: 375, top: y + 2, width: 720, height: 52 }, { fontSize: 22, color: INK });
    s.shapes.add({ geometry: "line", position: { left: 88, top: y + 72, width: 1030, height: 0 }, fill: "none", line: { style: "solid", fill: "#E2E8F0", width: 1 } });
  });
  addTakeaway(s, "建议汇报重点：围绕 frontier 解释 capacity、cost 和 reliability target 的关系。");
  addFooter(s, 13, 13, "Source: manuscript core body and current output figures");
  addNote(s, "最后邀请讨论：是否引入 critical load fraction？Figure 11/12 是否补做？目标期刊方法部分需要多详细？");

  for (const [i, slide] of pres.slides.items.entries()) {
    const file = path.join(PREVIEW_DIR, `slide-${String(i + 1).padStart(2, "0")}.png`);
    await writeBlob(file, await pres.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(PREVIEW_DIR, `slide-${String(i + 1).padStart(2, "0")}.layout.json`), await layout.text());
  }
  await writeBlob(path.join(OUT_DIR, "deck_montage.webp"), await pres.export({ format: "webp", montage: true, scale: 1 }));
  const pptx = await PresentationFile.exportPptx(pres);
  await pptx.save(FINAL_PPTX);

  const snapshot = await pres.inspect({ kind: "slide,textbox,shape,image,notes,layout", maxChars: 24000 });
  await fs.writeFile(path.join(OUT_DIR, "inspection.ndjson"), snapshot.ndjson);

  const qa = [
    "QA report",
    "Status: generated PPTX with slide previews and montage.",
    "Slide count: 13.",
    "Figures inserted: Figures 2-10 from SG_UTown outputs.",
    "Language: Chinese group-meeting presentation with English technical terms preserved.",
    "Self-review: one claim per slide; result slides use dominant original figure; visible text kept concise.",
    "Known limitation: slide previews are generated through artifact-tool rather than PowerPoint/LibreOffice rendering.",
  ].join("\n");
  await fs.writeFile(QA_REPORT, qa);

  console.log(FINAL_PPTX);
  console.log(path.join(OUT_DIR, "deck_montage.webp"));
}

build().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
