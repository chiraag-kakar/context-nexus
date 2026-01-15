from dataclasses import dataclass, field
from typing import Any, Literal
from contextlib import contextmanager
import time


@dataclass
class Span:
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def end(self):
        self.end_time = time.time()


class Tracer:
    """Simple tracer for debugging agent execution."""

    def __init__(
        self,
        backend: Literal["langsmith", "console", "none"] = "console",
        project: str = "context-nexus",
    ):
        self.backend = backend
        self.project = project
        self._spans: list[Span] = []

    @contextmanager
    def span(self, name: str):
        s = Span(name=name)
        self._spans.append(s)
        try:
            yield s
        finally:
            s.end()
            if self.backend == "console":
                print(f"[trace] {name}: {s.duration_ms:.1f}ms")

    def get_trace(self) -> list[dict]:
        return [{"name": s.name, "duration_ms": s.duration_ms, "attrs": s.attributes} for s in self._spans]
