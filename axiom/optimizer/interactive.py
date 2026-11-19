from __future__ import annotations

from .hardware import detect_hardware
from .planner import build_plan

AGENTS = {
    "1": "coding",
    "2": "cybersecurity",
    "3": "general assistant",
    "4": "reasoning",
    "5": "research",
    "6": "tool / agentic AI",
    "7": "creative / writing",
    "8": "custom",
}

PREFS = {"1": "speed", "2": "balanced", "3": "quality"}


def _choose(prompt, options):
    while True:
        value = input(prompt).strip()
        if value in options:
            return options[value]
        print("Please choose one of the listed numbers.")


def run_questionnaire(model):
    print("\n=== AXIOM AGENT OPTIMIZER ===\n")
    print("What are you building?")
    for k, v in AGENTS.items():
        print(f"  {k}) {v}")
    agent = _choose("\nSelect [1-8]: ", AGENTS)

    print("\nWhat matters most?")
    print("  1) Maximum speed")
    print("  2) Balanced")
    print("  3) Maximum quality")
    preference = _choose("Select [1-3]: ", PREFS)

    print("\nTarget performance:")
    print("  1) 5 tokens/sec")
    print("  2) 10 tokens/sec")
    print("  3) 20 tokens/sec")
    print("  4) Maximum possible")
    print("  5) Custom")
    target_choice = input("Select [1-5]: ").strip()
    targets = {"1": 5, "2": 10, "3": 20}
    if target_choice in targets:
        target = targets[target_choice]
    elif target_choice == "4":
        target = None
    else:
        target = float(input("Target tokens/sec: ").strip())

    context = int(input("\nContext length [default 4096]: ").strip() or "4096")
    hardware = detect_hardware()

    plan = build_plan(
        model,
        {
            "agent_type": agent,
            "preference": preference,
            "target_tps": target,
            "context": context,
        },
        hardware,
    )

    print("\n=== HARDWARE DETECTED ===")
    for key in ("os", "arch", "cpu_cores", "ram_gb", "gpu", "vram_mb", "backend"):
        print(f"{key:12}: {hardware.get(key)}")

    print("\n=== OPTIMIZED PROFILE ===")
    print(f"Agent type   : {plan['agent_type']}")
    print(f"Model        : {plan['model']}")
    print(f"Quantization : {plan['quantization']}")
    print(f"Context      : {plan['context']}")
    print(
        f"Target TPS   : {plan['target_tps'] if plan['target_tps'] else 'maximum possible'}"
    )
    print(f"Status       : {plan['status']}")
    print(f"NOTE         : {plan['note']}")
    return plan
