from __future__ import annotations

from typing import Iterable

from .schemas import SearchResult


CLIP_SUMMARY_SYSTEM_PROMPT = (
    "You summarize gameplay clips for local semantic search using only supplied evidence. "
    "Prefer concrete visible/spoken actions, outcomes, game entities, and player intent only when supported. "
    "Mention player position or movement, enemy position relative to the player, and teammate state or location "
    "only when the transcript, OCR, tags, audio events, or audio-video hints explicitly support it. "
    "Return valid JSON only, with no markdown or commentary."
)

DEEP_REASONING_SYSTEM_PROMPT = (
    "You answer questions about gameplay clips using only provided evidence. "
    "Do not infer people, teams, places, weapons, scores, or outcomes beyond the supplied summaries and transcripts. "
    "Cite clip ids for every factual claim. If evidence is thin or missing, say what is missing."
)


def build_summary_prompt(transcript: str, ocr_text: str = "", audio_events: Iterable[str] = ()) -> str:
    events = ", ".join(event for event in audio_events if event) or "none"
    return (
        f"Transcript:\n{transcript.strip() or '[no transcript]'}\n\n"
        f"OCR:\n{ocr_text.strip() or '[no OCR]'}\n\n"
        f"Audio events: {events}\n\n"
        "Return valid JSON only with this schema: "
        "{\"title\": string, \"summary\": string, \"key_moments\": [string], \"uncertainties\": [string]}. "
        "No markdown, no code fences, and no prose outside JSON. "
        "The summary must be one paragraph. Key moments must be directly supported by transcript, OCR, audio events, tags, "
        "or audio-video hints. "
        "Include supported details such as the player's position/movement, enemies on the "
        "left/right/front/behind/above, and teammate nearby/downed/pushing/comms cues. "
        "Do not infer or invent spatial/team details that are not in the transcript, OCR, "
        "audio events, tags, or audio-video hints. Put weak or missing evidence in uncertainties."
    )


def build_reasoning_prompt(question: str, results: Iterable[SearchResult]) -> str:
    evidence = []
    for index, result in enumerate(results, start=1):
        evidence.append(
            "\n".join(
                [
                    f"[{index}] clip_id={result.clip_id} score={result.score:.3f}",
                    f"title={result.metadata.title}",
                    f"game={result.metadata.game}",
                    f"summary={result.summary}",
                    f"transcript={result.transcript}",
                ]
            )
        )
    return (
        f"Question: {question.strip()}\n\n"
        f"Evidence:\n{chr(10).join(evidence) if evidence else '[no evidence]'}\n\n"
        "Answer directly and cite clip ids in brackets, for example [clip_id=12]. "
        "Use only the evidence above. If the evidence does not answer the question, say so and name the missing evidence."
    )
