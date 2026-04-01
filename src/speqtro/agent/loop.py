"""AgentLoop: wraps AgentRunner with trajectory persistence."""

import logging
import uuid
from dataclasses import dataclass, field

from speqtro.agent.runner import AgentRunner
from speqtro.agent.trajectory import Trajectory

logger = logging.getLogger("speqtro.loop")


class AgentLoop:
    """Multi-turn agent loop with trajectory memory."""

    def __init__(self, session):
        self.session = session
        self.trajectory = Trajectory()
        self._runner = AgentRunner(session, trajectory=self.trajectory)

    def run(self, query: str, context: dict | None = None, spectral_input=None):
        result = self._runner.run(query, context, spectral_input=spectral_input)
        if result:
            self.trajectory.add_turn(
                query=query,
                answer=result.summary or "",
                plan=result.plan,
            )
        return result
