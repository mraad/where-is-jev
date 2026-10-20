"""Probe hr98w/jev-visual on the lossless diagnostic patches, including option-order sensitivity."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chips", type=Path, default=Path("results/audit"))
    parser.add_argument("--model", type=Path, default=Path("checkpoints/Qwen3.5-0.8B-4bit"))
    parser.add_argument("--out", type=Path, default=Path("results/jev_probe.json"))
    parser.add_argument("--question", default=(
        "Is at least one above-ground circular storage tank visible in this overhead image? "
        "Include white, gray, dark, and floating-roof tanks, and partly visible tanks. "
        "Do not count rectangular buildings, vehicles, or empty circular foundations."
    ))
    args = parser.parse_args()
    images = sorted(args.chips.glob("*_rgb.png"))
    if not images:
        parser.error("No *_rgb.png chips found")
    if args.out.exists():
        parser.error("Output already exists; choose a new --out")
    from jev_visual import Request
    from jev_visual.engine import Engine

    engine = Engine(str(args.model))
    results = []
    question = args.question
    for image in images:
        result = engine.judge(Request(
            image=str(image.resolve()),
            state="An RGB satellite image of an industrial area. Judge visible evidence.",
            questions={
                "tank": {"type": "noul", "instructions": question},
                "reversed": {"type": "choice", "instructions": question,
                             "criteria": {"no": "No, no storage tank is visible.",
                                          "yes": "Yes, at least one storage tank is visible."}},
            },
        ))
        results.append({"patch": image.stem.removesuffix("_rgb"), "question": question, **result})
        print(json.dumps({"patch": results[-1]["patch"],
                          "p_yes": result["answers"]["tank"]["noul"],
                          "p_yes_reversed": result["answers"]["reversed"]["probabilities"]["yes"],
                          "elapsed_ms": result["metrics"]["elapsed_ms"]}), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
