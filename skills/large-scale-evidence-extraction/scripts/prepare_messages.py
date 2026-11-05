"""Prepare question-specific extraction or entity messages without calling a model."""

import argparse
import json
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
QUESTION_FIELDS = {
    "exposure", "outcome", "comparator", "population", "outcome_type",
    "contrast", "follow_up", "study_designs",
}


def nonempty_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value


def build_messages(question, article, stage, extraction=None, index=None):
    if not isinstance(question, dict):
        raise ValueError("Research question must be a JSON object.")
    unknown = set(question) - QUESTION_FIELDS
    if unknown:
        raise ValueError(f"Unknown question fields: {', '.join(sorted(unknown))}")
    for field in ("exposure", "outcome"):
        nonempty_text(question.get(field), field)
    for field, value in question.items():
        nonempty_text(value, field)
    if not isinstance(article, dict):
        raise ValueError("Article must be a JSON object with title and abstract.")
    source = {
        field: nonempty_text(article.get(field), field)
        for field in ("title", "abstract")
    }
    prompts = json.loads((SKILL_ROOT / "references/prompts.json").read_text())
    payload = {"research_question": question, "article": source}
    if stage == "extract":
        if extraction is not None or index is not None:
            raise ValueError("Extraction results and terminology index apply to the map stage.")
        payload["schema"] = prompts["schema"]
    elif stage == "map":
        if not isinstance(extraction, dict):
            raise ValueError("The map stage requires a populated extraction object.")
        if not isinstance(extraction.get("picos"), dict) or not isinstance(extraction.get("effects"), list):
            raise ValueError("Extraction must contain picos and effects; validate it before mapping.")
        concepts = []
        if index is not None:
            if not isinstance(index, dict) or not isinstance(index.get("concepts"), list):
                raise ValueError("Terminology index must contain a concepts array.")
            seen = set()
            for concept in index["concepts"]:
                if not isinstance(concept, dict):
                    raise ValueError("Each terminology concept must be an object.")
                identifier = nonempty_text(concept.get("id"), "concept id")
                label = nonempty_text(concept.get("label"), "concept label")
                if identifier in seen:
                    raise ValueError(f"Duplicate concept id: {identifier}")
                seen.add(identifier)
                concepts.append({"id": identifier, "label": label})
        payload.update(extracted=extraction, terminology_index=concepts)
    else:
        raise ValueError("Stage must be extract or map.")
    return [
        {"role": "system", "content": prompts[stage]},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("extract", "map"), required=True)
    parser.add_argument("--question", type=Path, required=True)
    parser.add_argument("--article", type=Path, required=True)
    parser.add_argument("--extraction", type=Path)
    parser.add_argument("--index", type=Path)
    args = parser.parse_args()
    try:
        question = json.loads(args.question.read_text())
        article = json.loads(args.article.read_text())
        extraction = json.loads(args.extraction.read_text()) if args.extraction else None
        index = json.loads(args.index.read_text()) if args.index else None
        messages = build_messages(question, article, args.stage, extraction, index)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(messages, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
