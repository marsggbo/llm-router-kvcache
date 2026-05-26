"""
Dataset loaders.

Serving benchmarks (routing + KV cache study):
  - MMLU        : structured QA, 57 subjects, fixed few-shot prefix → high KV reuse
  - WildBench   : diverse real-user tasks, task-type labels for routing
  - ShareGPT    : multi-turn conversations, long prefix accumulation

Router calibration / training (from RouteLLM HuggingFace org):
  - routellm/gpt4_judge_battles   (109k)  GPT-4-judged strong/weak labels
  - routellm/mmlu_battles         (1.53k) MMLU-specific routing labels
  - routellm/arena_battles_embeddings (55k) Chatbot Arena human preference
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class Request:
    prompt: str
    task_type: str          # e.g., "mmlu_math", "sharegpt", "wildbench_coding"
    expected_output: str = ""
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------

MMLU_TASK_GROUPS = {
    "stem": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning",
    ],
    "humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    ],
    "social_sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics",
        "high_school_psychology", "human_sexuality", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
    ],
    "other": [
        "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing",
        "medical_genetics", "miscellaneous", "nutrition",
        "professional_accounting", "professional_medicine", "virology",
    ],
}


def _subject_to_group(subject: str) -> str:
    for group, subjects in MMLU_TASK_GROUPS.items():
        if subject in subjects:
            return group
    return "other"


def load_mmlu(num_shots: int = 5, subjects: list = None, split: str = "test",
              max_samples: int = None) -> list[Request]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    all_subjects = subjects or sum(MMLU_TASK_GROUPS.values(), [])
    requests = []

    for subject in all_subjects:
        ds = load_dataset("cais/mmlu", subject, split=split)
        few_shot_ds = load_dataset("cais/mmlu", subject, split="dev")

        # Build few-shot prefix (shared within subject → high KV reuse potential)
        few_shot_examples = list(few_shot_ds)[:num_shots]
        prefix = f"The following are multiple choice questions about {subject.replace('_', ' ')}.\n\n"
        for ex in few_shot_examples:
            choices = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(ex["choices"]))
            prefix += f"Question: {ex['question']}\n{choices}\nAnswer: {chr(65 + ex['answer'])}\n\n"

        group = _subject_to_group(subject)
        for item in ds:
            choices = "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(item["choices"]))
            prompt = f"{prefix}Question: {item['question']}\n{choices}\nAnswer:"
            requests.append(Request(
                prompt=prompt,
                task_type=f"mmlu_{group}",
                expected_output=chr(65 + item["answer"]),
                metadata={"subject": subject, "group": group},
            ))

    if max_samples:
        random.shuffle(requests)
        requests = requests[:max_samples]

    return requests


# ---------------------------------------------------------------------------
# ShareGPT
# ---------------------------------------------------------------------------

def load_sharegpt(path: str, max_turns: int = 1, max_samples: int = None) -> list[Request]:
    """
    Load ShareGPT conversations. Uses the first `max_turns` user turns.
    Download from: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
    """
    data = json.loads(Path(path).read_text())
    requests = []
    for conv in data:
        turns = [m for m in conv.get("conversations", []) if m.get("from") == "human"]
        if not turns:
            continue
        prompt = turns[0]["value"]
        if max_turns > 1:
            # Build multi-turn prefix
            history = conv["conversations"][: max_turns * 2 - 1]
            parts = []
            for msg in history:
                role = "User" if msg["from"] == "human" else "Assistant"
                parts.append(f"{role}: {msg['value']}")
            prompt = "\n".join(parts) + "\nUser: " + turns[min(max_turns - 1, len(turns) - 1)]["value"]
        requests.append(Request(
            prompt=prompt,
            task_type="sharegpt",
            metadata={"conv_id": conv.get("id", "")},
        ))

    if max_samples:
        random.shuffle(requests)
        requests = requests[:max_samples]

    return requests


# ---------------------------------------------------------------------------
# WildBench
# ---------------------------------------------------------------------------

WILDBENCH_TASK_MAP = {
    # v2 primary_tag values (title-cased in dataset, matched lowercase)
    "coding & debugging": "coding",
    "math": "math",
    "reasoning": "reasoning",
    "planning": "reasoning",
    "information seeking": "information",
    "creative writing": "creative",
    "editing": "creative",
    "advice seeking": "advice",
    "data analysis": "data_analysis",
    "role playing": "creative",
    "brainstorming": "reasoning",
    # v1 legacy keys
    "reasoning & planning": "reasoning",
    "information/explanation seeking": "information",
    "creative tasks": "creative",
    "others": "other",
}

# Task-specific system prompts injected as shared prefix.
# All requests of the same task type share this prefix verbatim →
# the KV cache for this prefix is computed once and reused for every
# subsequent request of the same type on the same instance.
# This simulates real-world deployments (customer service, coding assistant,
# math tutor) where a fixed system prompt is prepended to every request.
WILDBENCH_SYSTEM_PROMPTS = {
    "wildbench_coding":       "You are an expert software engineer with deep knowledge of algorithms, data structures, and best practices across multiple programming languages. Help the user with their coding task clearly and concisely.",
    "wildbench_math":         "You are a mathematics tutor with expertise in algebra, calculus, statistics, and mathematical reasoning. Solve problems step by step, showing your work clearly.",
    "wildbench_reasoning":    "You are a logical reasoning and planning expert. Analyze problems systematically, consider multiple perspectives, and provide well-structured solutions.",
    "wildbench_information":  "You are a knowledgeable assistant with broad expertise across science, history, culture, and current events. Provide accurate, well-organized information.",
    "wildbench_creative":     "You are a creative writing and ideation expert. Help with storytelling, brainstorming, editing, and all forms of creative expression with imagination and style.",
    "wildbench_advice":       "You are a thoughtful advisor who provides balanced, practical guidance on personal, professional, and life decisions.",
    "wildbench_data_analysis":"You are a data analysis expert skilled in statistics, visualization, and extracting insights from complex datasets.",
    "wildbench_other":        "You are a helpful, harmless, and honest AI assistant. Answer questions accurately and helpfully.",
}


def load_wildbench(split: str = "test", max_samples: int = None,
                   use_system_prompt: bool = True) -> list[Request]:
    """
    Load WildBench v2 requests.

    use_system_prompt: if True, prepend a task-specific system prompt to each
      request. This creates a shared prefix within each task type, enabling
      KV cache reuse across same-type requests — simulating real deployment
      scenarios where a system prompt is always present.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    ds = load_dataset("allenai/WildBench", "v2", split=split)
    requests = []

    for item in ds:
        # v2 uses "conversation_input"; v1 uses "conversation"
        conversation = item.get("conversation_input") or item.get("conversation", [])
        if not conversation:
            continue
        user_content = conversation[0].get("content", "")
        if not user_content:
            continue
        raw_tag = (item.get("primary_tag") or "others").lower()
        task_type = "wildbench_" + WILDBENCH_TASK_MAP.get(raw_tag, "other")

        if use_system_prompt:
            system_prompt = WILDBENCH_SYSTEM_PROMPTS.get(task_type,
                            WILDBENCH_SYSTEM_PROMPTS["wildbench_other"])
            # Format: system prompt as a role-separated prefix
            prompt = f"System: {system_prompt}\n\nUser: {user_content}"
        else:
            prompt = user_content

        requests.append(Request(
            prompt=prompt,
            task_type=task_type,
            metadata={
                "session_id": item.get("session_id", ""),
                "tag": raw_tag,
                "has_system_prompt": use_system_prompt,
            },
        ))

    if max_samples:
        random.shuffle(requests)
        requests = requests[:max_samples]

    return requests


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def load_dataset_by_name(name: str, cfg: dict) -> list[Request]:
    dcfg = cfg["datasets"].get(name, {})
    if name == "mmlu":
        return load_mmlu(
            num_shots=dcfg.get("num_shots", 5),
            subjects=dcfg.get("subjects"),
            split=dcfg.get("split", "test"),
            max_samples=dcfg.get("max_samples"),
        )
    elif name == "sharegpt":
        return load_sharegpt(
            path=dcfg["path"],
            max_turns=dcfg.get("max_turns", 1),
            max_samples=dcfg.get("max_samples"),
        )
    elif name == "wildbench":
        return load_wildbench(
            split=dcfg.get("split", "test"),
            max_samples=dcfg.get("max_samples"),
            use_system_prompt=dcfg.get("use_system_prompt", True),
        )
    elif name == "routellm_gpt4":
        return load_routellm_battles(
            hf_dataset="routellm/gpt4_judge_battles",
            max_samples=dcfg.get("max_samples"),
        )
    elif name == "routellm_mmlu":
        return load_routellm_battles(
            hf_dataset="routellm/mmlu_battles",
            max_samples=dcfg.get("max_samples"),
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ---------------------------------------------------------------------------
# RouteLLM calibration datasets
# ---------------------------------------------------------------------------

def load_routellm_battles(hf_dataset: str, max_samples: int = None) -> list[Request]:
    """
    Load RouteLLM's HuggingFace battle datasets for router calibration/evaluation.

    Each record has a prompt and a binary label: whether the strong model won.
    Useful for calibrating the routing threshold on your own model pair.

    Datasets:
      routellm/gpt4_judge_battles  — 109k, GPT-4 as judge
      routellm/mmlu_battles        — 1.53k, MMLU subset
      routellm/arena_battles_embeddings — 55k, human preference
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    ds = load_dataset(hf_dataset, split="train", trust_remote_code=True)
    requests = []
    for item in ds:
        prompt = item.get("prompt", "") or item.get("question", "")
        if not prompt:
            continue
        # strong_win=1 means the strong model was judged better
        strong_win = int(item.get("strong_win", item.get("label", 0)))
        requests.append(Request(
            prompt=prompt,
            task_type="routellm_calibration",
            expected_output=str(strong_win),
            metadata={
                "source": hf_dataset,
                "strong_win": strong_win,
            },
        ))

    if max_samples:
        random.shuffle(requests)
        requests = requests[:max_samples]

    return requests
