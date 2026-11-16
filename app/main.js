// Interactive map of the tank experiment. Data comes from export_web.py (app/data/).
const LABELS = ["Contains a tank", "Partially contains a tank", "Does not contain a tank"];
const MODELS = ["qwen4b", "qwen32b", "nemotron12b", "nemotron30b"];
const KEV_MODELS = ["vision_linear", "kev_zero_shot", "kev_tuned_head", "unfiltered"];
const COLORS = { complete: [61, 220, 132], partial: [255, 176, 32], tp: [61, 220, 132], fp: [255, 48, 48],
  missed: [56, 200, 255], extent: [255, 225, 64], band: [255, 61, 217], dropped: [176, 176, 176] };
const $ = (id) => document.getElementById(id);

// --- Metrics (pure functions over feature properties) ---

// Same definition as compare_backbones.metrics: presence = complete or partial.
export function tileMetrics(rows, model) {
  const matrix = LABELS.map(() => [0, 0, 0]);
  for (const r of rows) matrix[LABELS.indexOf(r.truth)][LABELS.indexOf(r[model])] += 1;
  const tp = matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1];
  const fp = matrix[2][0] + matrix[2][1];
  const fn = matrix[0][2] + matrix[1][2];
  return { tp, fp, fn, accuracy: (matrix[0][0] + matrix[1][1] + matrix[2][2]) / rows.length,
    precision: tp / (tp + fp), recall: tp / (tp + fn) };
}

// Matching is one-to-one in descending score, so a score cut only truncates it: counts stay exact.
export function detectionMetrics(rows, threshold, labels) {
  const kept = rows.filter((r) => r.score >= threshold);
  const tp = kept.filter((r) => r.status === "TP").length;
  return { tp, fp: kept.length - tp, fn: labels - tp };
}

export function kevMetrics(rows, model, thresholds, labels) {
  const kept = rows.filter((r) => r.split === "test" && (model === "unfiltered" || r[model] >= thresholds[model]));
  const tp = kept.filter((r) => r.original_tp).length;
  return { tp, fp: kept.length - tp, fn: labels - tp };
}

const pct = (x) => (Number.isFinite(x) ? `${(100 * x).toFixed(1)}%` : "n/a");

function table(el, m) {
  const rows = [["True positives", m.tp], ["False positives", m.fp], ["False negatives", m.fn],
    ["Precision", pct(m.tp / (m.tp + m.fp))], ["Recall", pct(m.tp / (m.tp + m.fn))]];
  if (m.accuracy !== undefined) rows.push(["Three-way agreement", pct(m.accuracy)]);
  el.replaceChildren(...rows.map(([k, v]) => {
    const tr = document.createElement("tr");
    tr.append(Object.assign(document.createElement("td"), { textContent: k }),
      Object.assign(document.createElement("td"), { textContent: v }));
    return tr;
  }));
}

// Fails loudly if the browser numbers drift from the recorded Python results.
function selfCheck(tiles, detections, kev, summary) {
  const same = (a, b) => a.tp === b.tp && a.fp === b.fp && a.fn === b.fn;
  const failures = [];
  for (const model of MODELS) {
    const s = summary.tiles[model];
    const m = tileMetrics(tiles, model);
    if (!same(m, { tp: s.presence_tp, fp: s.presence_fp, fn: s.presence_fn }) || Math.abs(m.accuracy - s.accuracy) > 1e-9)
      failures.push(`tiles ${model}`);
  }
  if (!same(detectionMetrics(detections, 0, summary.detections.ground_truth_evaluated), summary.detections.iou_0_50))
    failures.push("detections");
  for (const model of KEV_MODELS) {
    if (!same(kevMetrics(kev, model, summary.kev.thresholds, summary.kev.test_reference_tanks),
      summary.kev.detection_evaluation[model])) failures.push(`kev ${model}`);
  }
  console.assert(failures.length === 0, "Metric self-check failed:", failures);
  return failures;
}

// --- Map ---

const fill = (rgb, alpha) => ({ type: "simple-fill", color: [...rgb, alpha], outline: { color: rgb, width: 1.2 } });
const outline = (rgb, width = 1.5, style = "solid") =>
  ({ type: "simple-fill", color: [0, 0, 0, 0], outline: { color: rgb, width, style } });

function tileRenderer(field) {
  return { type: "unique-value", field, legendOptions: { title: "Tile decision" }, uniqueValueInfos: [
    { value: LABELS[0], label: "Complete tank", symbol: fill(COLORS.complete, 0.18) },
    { value: LABELS[1], label: "Part of a tank", symbol: fill(COLORS.partial, 0.18) }] };
}

function kevRenderer(model, thresholds) {
  const kept = model === "unfiltered" ? "true" : `$feature.${model} >= ${thresholds[model]}`;
  return { type: "unique-value", legendOptions: { title: "Kev filter" },
    valueExpression: `IIf(${kept}, IIf($feature.original_tp == 1, "kept-tp", "kept-fp"), IIf($feature.original_tp == 1, "dropped-tp", "dropped-fp"))`,
    uniqueValueInfos: [
      { value: "kept-tp", label: "Kept, true tank", symbol: outline(COLORS.tp, 2) },
      { value: "kept-fp", label: "Kept, false alarm", symbol: outline(COLORS.fp, 2) },
      { value: "dropped-tp", label: "Dropped, true tank", symbol: outline(COLORS.tp, 2, "dash") },
      { value: "dropped-fp", label: "Dropped, false alarm", symbol: outline(COLORS.dropped, 1, "dash") }] };
}

async function main() {
  const [Map, Extent, MediaLayer, ImageElement, ExtentAndRotationGeoreference, GeoJSONLayer, kernel] = await Promise.all([
    "@arcgis/core/Map.js", "@arcgis/core/geometry/Extent.js", "@arcgis/core/layers/MediaLayer.js",
    "@arcgis/core/layers/support/ImageElement.js", "@arcgis/core/layers/support/ExtentAndRotationGeoreference.js",
    "@arcgis/core/layers/GeoJSONLayer.js", "@arcgis/core/kernel.js"].map((m) => $arcgis.import(m)));
  const load = (name) => fetch(`data/${name}`).then((r) => {
    if (!r.ok) throw new Error(`data/${name}: HTTP ${r.status}. Run export_web.py first.`);
    return r.json();
  });
  const [summary, tilesFc, detectionsFc, kevFc] = await Promise.all(
    ["summary.json", "tiles_512.geojson", "detections.geojson", "kev.geojson"].map(load));
  const props = (fc) => fc.features.map((f) => f.properties);
  const tiles = props(tilesFc).filter((r) => r.in_extent === 1);
  const detections = props(detectionsFc);
  const kev = props(kevFc);

  const s = summary.scene;
  const extent = new Extent({ xmin: s.xmin, ymin: s.ymin, xmax: s.xmax, ymax: s.ymax, spatialReference: { wkid: s.wkid } });
  const geojson = (name, title, options) => new GeoJSONLayer({ url: `data/${name}`, title, ...options });

  const scene = new MediaLayer({ title: "Aerial scene (2015)", listMode: "hide",
    source: [new ImageElement({ image: "data/scene.jpg", georeference: new ExtentAndRotationGeoreference({ extent }) })] });
  const tileLayer = geojson("tiles_512.geojson", "Gate tiles", {
    renderer: tileRenderer("qwen4b"), outFields: ["*"],
    popupTemplate: { title: "Tile {patch_id}", content: [{ type: "fields", fieldInfos: ["truth", ...MODELS].map((fieldName) => ({ fieldName })) }] } });
  const detectionLayer = geojson("detections.geojson", "SAM 3 detections", {
    outFields: ["*"],
    renderer: { type: "unique-value", field: "status", uniqueValueInfos: [
      { value: "TP", label: "True positive (IoU ≥ 0.5)", symbol: outline(COLORS.tp, 2) },
      { value: "FP", label: "False positive", symbol: outline(COLORS.fp, 1.5) }] },
    popupTemplate: { title: "Detection {id}: {status}", content: [{ type: "fields", fieldInfos: [
      { fieldName: "score", format: { places: 3 } }, { fieldName: "best_iou", label: "Best IoU with a label", format: { places: 2 } },
      { fieldName: "patch" }] }] } });
  const missedLayer = geojson("truth.geojson", "Missed labels", {
    definitionExpression: "status = 'FN'", renderer: { type: "simple", label: "Reference tank not matched", symbol: outline(COLORS.missed, 2.5) } });
  const labelLayer = geojson("truth.geojson", "Reference labels", { visible: false,
    renderer: { type: "simple", label: "Reference tank", symbol: outline([255, 255, 255], 1, "dash") } });
  const extentLayer = geojson("extent.geojson", "Evaluation extent", {
    renderer: { type: "simple", label: "Extent", symbol: outline(COLORS.extent, 2.5) } });
  const kevLayer = geojson("kev.geojson", "Kev pilot candidates", { visible: false, outFields: ["*"],
    definitionExpression: "split = 'test'", renderer: kevRenderer("vision_linear", summary.kev.thresholds),
    popupTemplate: { title: "Candidate {id}", content: [{ type: "fields", fieldInfos: ["original_tp", ...KEV_MODELS.slice(0, 3)].map((fieldName) => ({ fieldName, format: { places: 3 } })) }] } });
  const bandLayer = geojson("kev_band.geojson", "Kev test band", { visible: false,
    renderer: { type: "simple", label: "Spatial test band", symbol: outline(COLORS.band, 2, "dash") } });

  const mapEl = document.querySelector("arcgis-map");
  mapEl.map = new Map({ layers: [scene, tileLayer, labelLayer, detectionLayer, missedLayer, kevLayer, bandLayer, extentLayer] });
  mapEl.extent = extent;
  await mapEl.viewOnReady();
  if (matchMedia("(max-width: 600px)").matches) document.querySelector('arcgis-expand[slot="top-right"]').expanded = false;
  mapEl.constraints = { geometry: extent.clone().expand(1.5), minScale: 60000 };

  // --- Controls ---
  const tileQuery = () => {
    const model = $("tile-model").value;
    const parts = [];
    if ($("tile-extent").checked) parts.push("in_extent = 1");
    if ($("tile-disagree").checked && model !== "truth") parts.push(`${model} <> truth`, "in_extent = 1");
    tileLayer.renderer = tileRenderer(model);
    tileLayer.definitionExpression = parts.join(" AND ");
    $("tile-disagree").disabled = model === "truth";
    if (model === "truth") $("tile-metrics").replaceChildren();
    else table($("tile-metrics"), tileMetrics(tiles, model));
  };
  const scoreQuery = () => {
    const t = Number($("score").value);
    $("score-out").value = t.toFixed(2);
    detectionLayer.definitionExpression = `score >= ${t}`;
    table($("det-metrics"), detectionMetrics(detections, t, summary.detections.ground_truth_evaluated));
  };
  const kevQuery = () => {
    const model = $("kev-model").value;
    kevLayer.renderer = kevRenderer(model, summary.kev.thresholds);
    kevLayer.visible = bandLayer.visible = true;
    table($("kev-metrics"), kevMetrics(kev, model, summary.kev.thresholds, summary.kev.test_reference_tanks));
  };
  const minScore = Math.floor(100 * Math.min(...detections.map((r) => r.score))) / 100;
  Object.assign($("score"), { min: minScore, value: minScore });
  $("tile-model").addEventListener("change", tileQuery);
  $("tile-extent").addEventListener("change", tileQuery);
  $("tile-disagree").addEventListener("change", tileQuery);
  $("score").addEventListener("input", scoreQuery);
  $("kev-model").addEventListener("change", kevQuery);
  tileQuery();
  scoreQuery();
  table($("kev-metrics"), kevMetrics(kev, "vision_linear", summary.kev.thresholds, summary.kev.test_reference_tanks));

  const failures = selfCheck(tiles, detections, kev, summary);
  $("status").textContent = `ArcGIS Maps SDK from js.arcgis.com/5.2.2 (core build ${kernel.fullVersion}). ` + (failures.length
    ? `Metric self-check failed: ${failures.join(", ")}.` : "Metrics match the recorded results.");
  $("status").classList.toggle("error", failures.length > 0);
}

main().catch((error) => {
  console.error(error);
  $("status").textContent = error.message;
  $("status").classList.add("error");
});
