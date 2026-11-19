from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentProfile:
    number: int
    key: str
    name: str
    description: str
    context_tokens: int
    temperature: float
    preferred_quantization: str


PROFILES = (
    AgentProfile(
        1,
        "coding",
        "Coding Agent",
        "Code generation, debugging, repository work, and software engineering.",
        8192,
        0.20,
        "int4",
    ),
    AgentProfile(
        2,
        "reasoning",
        "Reasoning Agent",
        "Multi-step reasoning, planning, analysis, and difficult problem solving.",
        8192,
        0.25,
        "int4",
    ),
    AgentProfile(
        3,
        "research",
        "Research Agent",
        "Research, summarization, synthesis, and long-context analysis.",
        8192,
        0.30,
        "int4",
    ),
    AgentProfile(
        4,
        "assistant",
        "General Assistant",
        "General local assistant and everyday agent workloads.",
        4096,
        0.45,
        "int4",
    ),
    AgentProfile(
        5,
        "automation",
        "Automation Agent",
        "Tool calling, structured workflows, and task automation.",
        4096,
        0.20,
        "int4",
    ),
    AgentProfile(
        6,
        "math",
        "Math Agent",
        "Mathematics, symbolic reasoning, and numerical problem solving.",
        8192,
        0.15,
        "int4",
    ),
    AgentProfile(
        7,
        "writing",
        "Writing Agent",
        "Writing, editing, drafting, and style transformation.",
        4096,
        0.65,
        "int8",
    ),
    AgentProfile(
        8,
        "multilingual",
        "Multilingual Agent",
        "Multilingual conversation, translation, and mixed-language tasks.",
        4096,
        0.40,
        "int4",
    ),
)


def get_profile(number: int) -> AgentProfile:
    for profile in PROFILES:
        if profile.number == number:
            return profile
    raise ValueError("Agent profile must be a number from 1 to 8.")


def menu() -> str:
    lines = []
    for p in PROFILES:
        lines.append(f"{p.number}. {p.name} — {p.description}")
    return "\n".join(lines)
