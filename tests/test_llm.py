import os
import pytest
from ai_engine import generate_chat_response


@pytest.mark.skipif(not os.getenv("RUN_LLM_TESTS"), reason="LLM test disabled (set RUN_LLM_TESTS=1 to enable)")
def test_gemma_basic_response():
    answer = generate_chat_response("Hello", [])
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
