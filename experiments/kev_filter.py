"""Spatially held-out pilot: frozen Qwen vision linear probe versus tuned Kev pointer head.

Run with the Kev source on PYTHONPATH. Inputs are the full SAM 3 retest outputs.
No image or reference file is modified. All generated artifacts require a fresh --out.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from shapely.geometry import shape
from shapely.affinity import affine_transform

SPLITS = ("train", "calibration", "test")


def partition(left, right, cuts):
    """Assign only crops wholly in one band; straddling crops are excluded."""
    for i, (lo, hi) in enumerate(zip(cuts, cuts[1:])):
        if lo <= left and right <= hi:
            return SPLITS[i]
    return "buffer"


def select_threshold(scores, positive, minimum_recall=.95):
    """Choose the strictest threshold retaining >=95% calibration positives."""
    scores, positive = np.asarray(scores), np.asarray(positive, dtype=bool)
    if not positive.any():
        raise ValueError("Calibration needs positive examples")
    valid = [0.] + [float(t) for t in np.unique(scores)
                     if np.mean(scores[positive] >= t) >= minimum_recall]
    return max(valid)


def prepare(image, inputs, out):
    base = inputs / "candidate"
    raw = json.loads((base / "merged-retry/detections.geojson").read_text())["features"]
    evaluated = json.loads((base / "evaluation-retry/predictions.geojson").read_text())["features"]
    extent = shape(json.loads((base / "evaluation-retry/extent.geojson").read_text())["features"][0]["geometry"])
    with rasterio.open(image) as src:
        from pyproj import Transformer
        from shapely.ops import transform
        tf = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform
        inv = ~src.transform
        px_extent = affine_transform(transform(tf, extent), [inv.a, inv.b, inv.d, inv.e, inv.c, inv.f])
        x0, _, x1, _ = px_extent.bounds
        # Fixed geographic proportions chosen before reading any model results.
        cuts = [x0, x0 + .5*(x1-x0), x0 + .75*(x1-x0), x1]
        records = []
        (out / "crops").mkdir()
        for feature in evaluated:
            p = feature["properties"]
            ident = p["id"]
            x, y, xx, yy = raw[ident]["properties"]["pixel_box"]
            side = max(128, int(np.ceil(1.5 * max(xx-x, yy-y))))
            cx, cy = (x+xx)/2, (y+yy)/2
            left, top = int(cx-side/2), int(cy-side/2)
            split = partition(left, left+side, cuts)
            if left < 0 or top < 0 or left+side > src.width or top+side > src.height:
                split = "buffer"
            label = 1 if p["best_iou"] >= .5 else 0 if p["best_iou"] == 0 else -1
            record = dict(id=ident, split=split, label=label, best_iou=p["best_iou"],
                          original_tp=p["status"] == "TP", crop=[left, top, side, side],
                          matched_truth_id=p["matched_truth_id"])
            if split != "buffer":
                rgb = src.read([1, 2, 3], window=Window(left, top, side, side),
                               out_shape=(3, 224, 224), resampling=Resampling.bilinear)
                Image.fromarray(rgb.transpose(1, 2, 0)).save(out / "crops" / f"{ident}.png")
            records.append(record)
    (out / "records.json").write_text(json.dumps(dict(cuts=cuts, records=records), indent=2))
    return records


def representations(records, out):
    from huggingface_hub import HfApi, snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoImageProcessor
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    from kev.checkpoint import Checkpoint, LoadOptions
    from kev.api import SystemOneRequest, to_record
    from kev.model import rows_of

    revision = HfApi().model_info("jaredpalmer/kev-0.8b", revision="v1.0").sha
    ck = Checkpoint("jaredpalmer/kev-0.8b@" + revision)
    tok, model = ck.load("cuda", LoadOptions(dtype=torch.bfloat16, fused=False,
                                            cuda_graphs=False, backend="torch"))
    model.eval()
    model.requires_grad_(False)
    cfg = AutoConfig.from_pretrained(ck.meta.base, revision=ck.meta.base_revision)
    proc = AutoImageProcessor.from_pretrained(ck.meta.base, revision=ck.meta.base_revision)
    weights = snapshot_download(ck.meta.base, revision=ck.meta.base_revision,
                                allow_patterns=["*.safetensors"])
    visual = Qwen3_5VisionModel._from_config(cfg.vision_config)
    state = {}
    for path in Path(weights).glob("*.safetensors"):
        with safe_open(path, framework="pt") as stream:
            state.update({k.removeprefix("model.visual."): stream.get_tensor(k)
                          for k in stream.keys() if k.startswith("model.visual.")})
    visual.load_state_dict(state, strict=True)
    del state
    visual.to("cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    req = SystemOneRequest(state="RGB overhead imagery. Judge the candidate at the center.",
        questions={"tank": {"type": "choice", "instructions":
            "Is the central candidate an industrial storage tank? Reject roofs, roads, pipes and shadows.",
            "criteria": {"no": "Not a storage tank", "yes": "A storage tank"}}})
    rec, _ = to_record(req)
    prefix, _, rows = rows_of(model.encode(tok, rec))
    row = rows[0]
    embedding = model.lm.get_input_embeddings()
    zs, vision, decisions, options = [], [], [], []
    started = time.perf_counter()
    with torch.no_grad():
        for n, record in enumerate(records):
            with Image.open(out / "crops" / f'{record["id"]}.png') as image:
                batch = proc(images=[image], return_tensors="pt")
            grid = batch["image_grid_thw"].to("cuda")
            feats = visual(batch["pixel_values"].to("cuda", torch.bfloat16),
                           grid_thw=grid).pooler_output
            prefix_ids = prefix + [cfg.vision_start_token_id]
            suffix_ids = [cfg.vision_end_token_id] + row["ids"]
            e0 = embedding(torch.tensor(prefix_ids, device="cuda"))
            e1 = embedding(torch.tensor(suffix_ids, device="cuda"))
            t, h, w = grid[0].tolist()
            h, w = h//cfg.vision_config.spatial_merge_size, w//cfg.vision_config.spatial_merge_size
            axes = torch.meshgrid(*(torch.arange(v, device="cuda") for v in (t,h,w)), indexing="ij")
            pimage = torch.stack(axes).reshape(3,-1) + len(prefix_ids)
            assert feats.shape[0] == pimage.shape[1]
            after = len(prefix_ids) + max(t,h,w)
            pos = torch.cat([torch.arange(len(prefix_ids), device="cuda").expand(3,-1),
                             pimage, (torch.arange(len(suffix_ids), device="cuda")+after).expand(3,-1)], 1)
            emb = torch.cat([e0,feats.to(e0.dtype),e1])
            pos = torch.cat([torch.arange(len(emb),device="cuda")[None],pos])[:,None]
            hidden = model.lm(inputs_embeds=emb[None], position_ids=pos).last_hidden_state[0].float()
            offset = len(emb)-len(row["ids"])
            d = hidden[offset+row["decide"]]
            o = hidden[[offset+i for i in row["opts"]]]
            zs.append(model.head(d,o).cpu())
            vision.append(feats.float().mean(0).cpu())
            decisions.append(d.cpu()); options.append(o.cpu())
            print(f"features {n+1}/{len(records)}", flush=True)
    torch.cuda.synchronize()
    head = copy.deepcopy(model.head).cpu()
    head.temperature = 1.
    result = dict(vision=torch.stack(vision), decisions=torch.stack(decisions),
                  options=torch.stack(options), zero_logits=torch.stack(zs))
    torch.save(result, out/"features.pt")
    torch.save(head.state_dict(),out/"initial_head.pt")
    (out/"provenance.json").write_text(json.dumps(dict(kev_revision=revision,
        base=ck.meta.base, base_revision=ck.meta.base_revision, torch=torch.__version__,
        feature_seconds=time.perf_counter()-started, gpu=torch.cuda.get_device_name()),indent=2))
    return result, head


def evaluate_detections(inputs, image, gdb, out):
    """Re-match held-out predictions; report all reference tanks in the test band."""
    from evaluate import project, read_layer, restrict, overlap_pairs, match
    from shapely.ops import unary_union
    from sklearn.metrics import roc_auc_score

    report = json.loads((out / "report.json").read_text())
    scores = json.loads((out / "scores.json").read_text())
    cuts = json.loads((out / "records.json").read_text())["cuts"]
    reference, crs = read_layer(gdb, "Tanks")
    extents, extent_crs = read_layer(gdb, "Extent")
    extent = unary_union([project(g, extent_crs, crs) for _, g in extents])
    truth, _ = restrict([dict(id=i, geometry=g) for i, g in reference], extent)
    with rasterio.open(image) as src:
        inverse = ~src.transform
        test_truth = [r for r in truth if cuts[2] <= (
            inverse * project(next(g for i, g in reference if i == r["id"]).centroid,
                              crs, src.crs).coords[0])[0] <= cuts[3]]
    predictions = json.loads((inputs / "candidate/evaluation-retry/predictions.geojson").read_text())["features"]
    by_id = {f["properties"]["id"]: dict(f["properties"],
             geometry=project(shape(f["geometry"]), "EPSG:4326", crs)) for f in predictions}
    selected = [r for r in scores if r["split"] == "test"]
    metrics = {}
    for name in ["unfiltered", *report["models"]]:
        threshold = report["models"][name]["threshold"] if name != "unfiltered" else None
        chosen = [by_id[r["id"]] for r in selected if threshold is None or r[name] >= threshold]
        result = match(chosen, test_truth, overlap_pairs(chosen, test_truth), .5, True)
        metrics[name] = {k: v for k, v in result.items() if k != "matches"}
    report["detection_evaluation"] = metrics
    report["test_reference_tanks"] = len(test_truth)
    report["training_examples"] = sum(r["split"] == "train" and r["label"] >= 0 for r in scores)
    report["buffer_excluded"] = sum(r["split"] == "buffer" for r in
        json.loads((out / "records.json").read_text())["records"])
    # A geometric overlap can connect separate crops to the same reference object.
    owners = {}
    all_truth = [dict(id=i, geometry=g) for i,g in reference]
    all_predictions = [by_id[r["id"]] for r in scores]
    for pair in overlap_pairs(all_predictions, all_truth):
        owners.setdefault(pair["truth_index"], set()).add(scores[pair["prediction_index"]]["split"])
    crossing = [all_truth[i]["id"] for i, split in owners.items() if len(split) > 1]
    report["reference_objects_crossing_partitions"] = crossing
    if crossing:
        raise ValueError(f"Reference objects cross splits: {crossing}; pilot is not independent")
    for name in report["models"]:
        report["models"][name]["test_auc_original_tp"] = float(roc_auc_score(
            [r["original_tp"] for r in selected], [r[name] for r in selected]))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def run(args):
    if args.out.exists():
        raise ValueError("Use a fresh output directory")
    args.out.mkdir(parents=True)
    torch.manual_seed(42); np.random.seed(42)
    records = prepare(args.image, args.inputs, args.out)
    records = [r for r in records if r["split"] != "buffer"]
    train = np.array([r["split"]=="train" and r["label"]>=0 for r in records])
    cal = np.array([r["split"]=="calibration" for r in records])
    test = np.array([r["split"]=="test" for r in records])
    labels = np.array([max(0,r["label"]) for r in records])
    for name, mask in (("train",train),("calibration",cal),("test",test)):
        print(name, int(mask.sum()), "clear positive",int(labels[mask].sum()), flush=True)
        if len(set(labels[mask])) != 2:
            raise ValueError(f"{name} needs both classes; do not move split after reading test scores")
    f, head = representations(records,args.out)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(f["vision"][train])
    x = scaler.transform(f["vision"])
    linear = LogisticRegression(C=1., max_iter=2000, class_weight="balanced",random_state=42)
    linear.fit(x[train],labels[train])
    scores = {"vision_linear":linear.predict_proba(x)[:,1],
              "kev_zero_shot":f["zero_logits"].softmax(-1)[:,1].numpy()}
    head.requires_grad_(True).train()
    optimizer = torch.optim.AdamW(head.parameters(),lr=2e-5)
    target = torch.tensor(labels[train],dtype=torch.long)
    # Fixed small training budget; no test-driven epoch or hyperparameter selection.
    for epoch in range(40):
        optimizer.zero_grad()
        logits = torch.stack([head(d,o) for d,o in zip(f["decisions"][train],f["options"][train])])
        loss = torch.nn.functional.cross_entropy(logits,target)
        loss.backward(); optimizer.step()
    head.eval()
    with torch.no_grad():
        scores["kev_tuned_head"] = torch.stack([head(d,o) for d,o in zip(
            f["decisions"],f["options"])]).softmax(-1)[:,1].numpy()
    torch.save(head.state_dict(),args.out/"tuned_head.pt")
    np.savez(args.out/"linear_probe.npz",mean=scaler.mean_,scale=scaler.scale_,
             coef=linear.coef_,intercept=linear.intercept_)
    report = {"scope":"single-scene spatial pilot; geometry-derived labels, no independent human review",
              "training":"40 steps, pointer head only; vision encoder, LoRA and language backbone frozen",
              "splits":{s:sum(r["split"]==s for r in records) for s in SPLITS}, "models":{}}
    for name, score in scores.items():
        threshold = select_threshold(score[cal],np.array([r["original_tp"] for r in records])[cal])
        kept = score>=threshold
        tp = np.array([r["original_tp"] for r in records])
        report["models"][name] = dict(threshold=threshold,
            test_candidates=int(test.sum()),test_kept=int((test&kept).sum()),
            baseline_tp=int((test&tp).sum()),retained_tp=int((test&tp&kept).sum()),
            baseline_fp=int((test&~tp).sum()),retained_fp=int((test&~tp&kept).sum()))
        for r,prob in zip(records,score): r[name]=float(prob)
    (args.out/"scores.json").write_text(json.dumps(records,indent=2))
    (args.out/"report.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(evaluate_detections(args.inputs, args.image, args.gdb, args.out),indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image",type=Path,required=True)
    p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--gdb",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    run(p.parse_args())
