"""Trajectory: session memory across queries in interactive mode."""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Turn:
    query: str
    answer: str
    tools_used: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class Trajectory:
    """Session memory: records turns and provides context for the planner."""

    def __init__(self, max_turns: int = 20, session_id: str = None):
        self.turns: list[Turn] = []
        self.max_turns = max_turns
        self.session_id = session_id
        self.created_at = time.time()

    def add_turn(self, query: str, answer: str, plan=None):
        tools_used = []
        if plan and hasattr(plan, "steps"):
            tools_used = [s.tool for s in plan.steps if s.status == "completed"]
        self.turns.append(Turn(query=query, answer=answer, tools_used=tools_used))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def context_for_planner(self) -> str:
        if not self.turns:
            return ""
        recent = self.turns[-5:]
        lines = ["## Session Context (prior queries this session)", ""]
        for i, turn in enumerate(recent, 1):
            tools_str = ", ".join(turn.tools_used) if turn.tools_used else "none"
            answer_preview = turn.answer[:200] + "..." if len(turn.answer) > 200 else turn.answer
            lines.append(f"**Turn {i}**: {turn.query}")
            lines.append(f"  Tools: {tools_str}")
            lines.append(f"  Finding: {answer_preview}")
            lines.append("")
        return "\n".join(lines)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            meta = {"type": "meta", "session_id": self.session_id, "created_at": self.created_at}
            f.write(json.dumps(meta) + "\n")
            for turn in self.turns:
                f.write(json.dumps({"type": "turn", **asdict(turn)}) + "\n")

    @staticmethod
    def sessions_dir() -> Path:
        d = Path.home() / ".speqtro" / "sessions"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @classmethod
    def list_sessions(cls) -> list[dict]:
        sessions_dir = cls.sessions_dir()
        sessions = []
        for path in sessions_dir.glob("*.jsonl"):
            try:
                with open(path) as f:
                    first_line = f.readline().strip()
                    if first_line:
                        meta = json.loads(first_line)
                        if meta.get("type") == "meta":
                            meta["path"] = str(path)
                            sessions.append(meta)
            except (json.JSONDecodeError, OSError):
                continue
        sessions.sort(key=lambda s: s.get("created_at", 0), reverse=True)
        return sessions
