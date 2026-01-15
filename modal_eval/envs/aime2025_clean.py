import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer, load_example_dataset


def extract_answer_line(text: str) -> str:
    if not text:
        return text

    # Prefer boxed answers if they appear in the output.
    if "\\boxed{" in text:
        return extract_boxed_answer(text)

    lowered = text.lower()
    idx = lowered.rfind("answer:")
    if idx == -1:
        return text

    remainder = text[idx + len("answer:") :]
    lines = remainder.splitlines()
    if not lines:
        return remainder.strip()

    line = lines[0].strip()
    if not line:
        for candidate in lines[1:]:
            candidate = candidate.strip()
            if candidate:
                line = candidate
                break

    if line.startswith("$") and line.endswith("$") and len(line) >= 2:
        line = line[1:-1].strip()

    while line and line[-1] in ".,":  # drop trailing punctuation
        line = line[:-1].rstrip()

    return line


def load_environment(system_prompt: str | None = None, **kwargs) -> vf.Environment:
    eval_dataset = load_example_dataset("aime2025")
    parser = vf.MaybeThinkParser(extract_answer_line)
    rubric = vf.MathRubric(parser=parser)
    return vf.SingleTurnEnv(
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )
