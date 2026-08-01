"""Command-line utilities for transformers-xpu-helper."""

from __future__ import annotations

import argparse
import json
import sys


def _info_payload() -> dict:
    from .config import ultra_255h_config
    from .env import apply_runtime_env, describe_env
    from .hardware import detect_device
    from .memory import estimate_budget, format_bytes

    info = detect_device(prefer="xpu", profile_hint="255H")
    cfg = ultra_255h_config()
    apply_runtime_env(cfg, info.profile)
    budget = estimate_budget(cfg, info)

    return {
        "device": {
            "kind": info.kind.value,
            "device": info.device,
            "name": info.name,
            "torch_version": info.torch_version,
            "xpu_available": info.xpu_available,
            "total_memory": format_bytes(info.total_memory_bytes or 0)
            if info.total_memory_bytes
            else None,
        },
        "profile": {
            "name": info.profile.name,
            "codename": info.profile.codename,
            "gpu": info.profile.gpu_name,
            "xe_cores": info.profile.xe_cores,
            "shared_memory": info.profile.shared_memory,
            "preferred_amp": info.profile.preferred_amp.value,
            "supports_grad_scaler": info.profile.supports_grad_scaler,
            "notes": list(info.profile.notes),
        },
        "config": cfg.to_dict(),
        "memory_budget": {
            "total_ram": format_bytes(budget.total_ram_bytes),
            "host_reserve": format_bytes(budget.host_reserve_bytes),
            "trainable": format_bytes(budget.trainable_bytes),
            "fraction": budget.memory_fraction,
        },
        "env": describe_env(),
        "install_hint": (
            "pip install torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/xpu"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="xpu-helper-info",
        description="Show Intel XPU / Core Ultra 7 255H training helper diagnostics.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    args = parser.parse_args(argv)

    payload = _info_payload()
    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    device = payload["device"]
    profile = payload["profile"]
    budget = payload["memory_budget"]
    print("transformers-xpu-helper")
    print(f"  device     : {device['device']} ({device['name']})")
    print(f"  xpu ready  : {device['xpu_available']}")
    print(f"  torch      : {device['torch_version']}")
    print(f"  profile    : {profile['name']} / {profile['gpu']}")
    amp = profile["preferred_amp"]
    scaler = profile["supports_grad_scaler"]
    print(f"  amp        : {amp} (grad_scaler={scaler})")
    print(f"  memory     : trainable {budget['trainable']} of {budget['total_ram']}")
    print("  notes:")
    for note in profile["notes"]:
        print(f"    - {note}")
    print(f"  install    : {payload['install_hint']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
