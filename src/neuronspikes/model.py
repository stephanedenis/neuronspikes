from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SpikingModel:
    """Prototype minimal d'un modèle SNN.

    Ce squelette sera remplacé par la dynamique détaillée ultérieurement.
    """

    threshold: float = 1.0
    decay: float = 0.95
    state: float = 0.0
    spikes: List[int] = field(default_factory=list)

    def step(self, input_current: float) -> int:
        self.state = self.state * self.decay + input_current
        spike = 1 if self.state >= self.threshold else 0
        if spike:
            self.state = 0.0
        self.spikes.append(spike)
        return spike

    def run(self, inputs: List[float]) -> List[int]:
        return [self.step(value) for value in inputs]
