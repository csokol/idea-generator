"""Data classes for signals, evidence, and opportunities."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Signal:
    pain: str
    workaround: str
    desired_outcome: str
    segment: str
    severity: int  # 1-5
    willingness_to_pay: int  # 1-5
    keywords: list[str]
    doc_id: str
    url: str

    @property
    def normalized_pain(self) -> str:
        return self.pain.strip().lower()


@dataclass
class Opportunity:
    title: str
    summary: str
    signals: list[Signal] = field(default_factory=list)
    score: float = 0.0
    confidence: str = "low"
    market_context: str = ""

    @property
    def evidence_count(self) -> int:
        return len(self.signals)

    @property
    def unique_urls(self) -> set[str]:
        return {s.url for s in self.signals}

    @property
    def avg_severity(self) -> float:
        if not self.signals:
            return 0.0
        return sum(s.severity for s in self.signals) / len(self.signals)

    @property
    def avg_willingness_to_pay(self) -> float:
        if not self.signals:
            return 0.0
        return sum(s.willingness_to_pay for s in self.signals) / len(self.signals)


class EvidenceStore:
    """In-memory store tracking signals keyed by normalized pain, with dedup."""

    def __init__(self) -> None:
        self._signals: dict[str, list[Signal]] = {}
        self._seen_doc_ids: set[str] = set()

    def add(self, signal: Signal) -> bool:
        """Add signal if not duplicate. Returns True if added."""
        if signal.doc_id in self._seen_doc_ids:
            return False
        self._seen_doc_ids.add(signal.doc_id)
        key = signal.normalized_pain
        self._signals.setdefault(key, []).append(signal)
        return True

    @property
    def total_signals(self) -> int:
        return sum(len(v) for v in self._signals.values())

    @property
    def pain_groups(self) -> dict[str, list[Signal]]:
        return dict(self._signals)

    def all_signals(self) -> list[Signal]:
        return [s for group in self._signals.values() for s in group]
