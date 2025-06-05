import uuid
from attrs import define, field

from typing import Any, Callable, TypedDict


class Message(TypedDict):
  role: str
  content: str


AnyState = dict[str, Any]


@define
class Milestone:
  description: str
  expected_state: AnyState


@define
class TestCase:
  task_id: str = field(factory=lambda: str(uuid.uuid4()))
  description: str
  domain: str
  level: int
  initial_messages: list[Message]
  initial_state: AnyState
  expected_outcome: str
  expected_state: AnyState
  available_tools: list[Callable]
  milestones: list[Milestone]
