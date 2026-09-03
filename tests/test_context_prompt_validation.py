"""v4.17.1 — /context refuses an empty or whitespace-only prompt.

Before this, `prompt: ""` embedded fine and explore mode served a chunk
against nothing. The request model is the door; FastAPI turns a pydantic
ValidationError into 422, so the contract is proven at the model.
"""
import pytest
from pydantic import ValidationError

from agentb.server import ContextRequest


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
def test_blank_prompt_is_rejected(prompt):
    with pytest.raises(ValidationError) as exc:
        ContextRequest(prompt=prompt, agent_id="t")
    assert "prompt" in str(exc.value)


def test_real_prompt_still_accepted():
    req = ContextRequest(prompt=" what did we decide ", agent_id="t")
    assert req.prompt == " what did we decide "   # not stripped, only checked
