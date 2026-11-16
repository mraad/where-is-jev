// Step-by-step map of the tank experiment. Data comes from export_web.py (app/data/).
// Each step shows only its own layers, so every click changes what is on the map.
const LABELS = ["Contains a tank", "Partially contains a tank", "Does not contain a tank"]; // patch_decisions.LABELS
const COLORS = { extent: [255, 225, 64], complete: [61, 220, 132], partial: [255, 176, 32],
  outline: [255, 255, 255], tp: [61, 220, 132], fp: [255, 48, 48], missed: [56, 200, 255] };
const $ = (id) => document.getElementById(id);

const fill = (rgb, alpha) => ({ type: "simple-fill", color: [...rgb, alpha], outline: { color: rgb, width: 1 } });
const outline = (rgb, width) => ({ type: "simple-fill", color: [0, 0, 0, 0], outline: { color: rgb, width } });

// The story. Each step: which layers are visible, what to say, and a color key with counts.
function steps(n) {
  return [
    { title: "One aerial photo", layers: ["extent"],
      body: `<p>This is one aerial photo from 2015. Each pixel is 0.5 m.</p>
        <p>Inside the yellow box, a person outlined every storage tank by hand: ${n.labels} tanks.
        We use those outlines only to check the result at the end.</p>
        <p>The goal: find the tanks with models that were never trained on tanks.</p>`,
      key: [[COLORS.extent, "", "Area with hand-labeled tanks"]] },
    { title: "Step 1: Ask a model about each tile", layers: ["extent", "tiles"],
      body: `<p>We cut the photo into ${n.tiles} square tiles of 256 m. Each tile overlaps its neighbors by half,
        so a tank cut by one tile edge is usually whole in a neighbor tile.</p>
        <p>For each tile, a vision-language model (${n.model}) answers one question:
        does this tile contain a tank, part of a tank, or no tank?</p>
        <p>${n.skipped} tiles got “no tank” and are not shown. Only the colored tiles go to the next step.</p>`,
      key: [[COLORS.complete, n.complete, "Contains a tank"], [COLORS.partial, n.partial, "Part of a tank"]] },
    { title: "Step 2: Outline the tanks", layers: ["extent", "detections"],
      body: `<p>A segmentation model (SAM 3) gets each kept tile and the text “round storage tanks”.
        It draws an outline around every object that matches.</p>
        <p>Tiles overlap, so one tank is often outlined several times. We remove those duplicates.</p>
        <p>${n.detections} outlines are inside the yellow box.</p>`,
      key: [[COLORS.outline, n.detections, "Outline from SAM 3"]] },
    { title: "Step 3: Check against the hand labels", layers: ["extent", "detections", "missed"], scored: true,
      body: `<p>Now we compare each outline with the hand-drawn tanks.
        An outline counts as correct when it covers the same tank closely (IoU ≥ 0.5).</p>
        <p>The method found <strong>${n.tp} of ${n.labels}</strong> tanks (${n.recall}).
        It also drew ${n.fp} extra outlines, so ${n.precision} of all outlines are tanks.</p>
        <p class="muted">Many extra outlines are other round objects, or tanks that nobody labeled;
        they were not checked one by one. All ${n.labels} tanks are touched by some outline; the
        ${n.fn} missed ones fail only the shape test. One photo only; nothing was tuned to the labels.</p>`,
      key: [[COLORS.tp, n.tp, "Correct: matches a labeled tank"], [COLORS.fp, n.fp, "Extra: does not match a labeled tank"],
        [COLORS.missed, n.fn, "Missed labeled tank"]] },
  ];
}

const pct = (x) => `${(100 * x).toFixed(1)}%`;

async function main() {
  const modules = Promise.all([
    "@arcgis/core/Map.js", "@arcgis/core/geometry/Extent.js", "@arcgis/core/layers/MediaLayer.js",
    "@arcgis/core/layers/support/ImageElement.js", "@arcgis/core/layers/support/ExtentAndRotationGeoreference.js",
    "@arcgis/core/layers/GeoJSONLayer.js"].map((m) => $arcgis.import(m)));
  const response = await fetch("data/summary.json");
  if (!response.ok) throw new Error(`data/summary.json: HTTP ${response.status}. Run export_web.py first.`);
  const summary = await response.json();
  const [ArcGISMap, Extent, MediaLayer, ImageElement, ExtentAndRotationGeoreference, GeoJSONLayer] = await modules;

  // All counts are the recorded Python results, so the page cannot drift from them.
  const gate = summary.tiles[summary.gate];
  const tileCounts = JSON.parse(gate.all_tile_counts);
  const { tp, fp, fn } = summary.detections.iou_0_50;
  const n = { labels: summary.detections.ground_truth_evaluated, model: gate.checkpoint.split("/").pop(), tiles: gate.tiles,
    complete: tileCounts[LABELS[0]], partial: tileCounts[LABELS[1]], skipped: tileCounts[LABELS[2]],
    detections: tp + fp, tp, fp, fn, recall: pct(tp / (tp + fn)), precision: pct(tp / (tp + fp)) };

  const s = summary.scene;
  const extent = new Extent({ xmin: s.xmin, ymin: s.ymin, xmax: s.xmax, ymax: s.ymax, spatialReference: { wkid: s.wkid } });
  const geojson = (name, options) => new GeoJSONLayer({ url: `data/${name}`, visible: false, ...options });
  const plain = { type: "simple", symbol: outline(COLORS.outline, 1.5) };
  const scored = { type: "unique-value", field: "status", uniqueValueInfos: [
    { value: "TP", symbol: outline(COLORS.tp, 2) }, { value: "FP", symbol: outline(COLORS.fp, 1.5) }] };
  const layers = {
    tiles: geojson("tiles_512.geojson", { definitionExpression: `${summary.gate} <> '${LABELS[2]}'`,
      renderer: { type: "unique-value", field: summary.gate, uniqueValueInfos: [
        { value: LABELS[0], symbol: fill(COLORS.complete, 0.15) },
        { value: LABELS[1], symbol: fill(COLORS.partial, 0.15) }] } }),
    detections: geojson("detections.geojson", { renderer: plain }),
    missed: geojson("truth.geojson", { definitionExpression: "status = 'FN'", renderer: { type: "simple", symbol: outline(COLORS.missed, 3) } }),
    extent: geojson("extent.geojson", { renderer: { type: "simple", symbol: outline(COLORS.extent, 2.5) } }),
  };
  const scene = new MediaLayer({
    source: [new ImageElement({ image: "data/scene.jpg", georeference: new ExtentAndRotationGeoreference({ extent }) })] });

  const mapEl = document.querySelector("arcgis-map");
  mapEl.map = new ArcGISMap({ layers: [scene, ...Object.values(layers)] });
  mapEl.extent = extent;
  await mapEl.viewOnReady();
  mapEl.constraints = { geometry: extent.clone().expand(1.5), minScale: 60000 };

  const story = steps(n);
  let at = 0;
  const show = () => {
    const step = story[at];
    for (const [name, layer] of Object.entries(layers)) layer.visible = step.layers.includes(name);
    layers.detections.renderer = step.scored ? scored : plain;
    $("count").textContent = `${at + 1} of ${story.length}`;
    $("title").textContent = step.title;
    $("body").innerHTML = step.body;
    $("key").replaceChildren(...step.key.map(([rgb, count, label]) => {
      const li = document.createElement("li");
      li.append(Object.assign(document.createElement("span"), { className: "swatch", style: `border-color:rgb(${rgb})` }),
        Object.assign(document.createElement("b"), { textContent: count }), label);
      return li;
    }));
    $("back").disabled = at === 0;
    $("next").disabled = at === story.length - 1;
  };
  $("back").addEventListener("click", () => { at -= 1; show(); });
  $("next").addEventListener("click", () => { at += 1; show(); });
  show();
  $("status").textContent = "";
}

main().catch((error) => {
  console.error(error);
  $("status").textContent = error.message;
  $("status").classList.add("error");
});
