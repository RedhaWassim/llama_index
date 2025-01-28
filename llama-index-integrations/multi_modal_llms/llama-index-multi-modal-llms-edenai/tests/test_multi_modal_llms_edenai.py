from llama_index.core.multi_modal_llms.base import MultiModalLLM
from llama_index.multi_modal_llms.edenai import EdenaiMultiModal
import pytest
from unittest.mock import patch, MagicMock
from llama_index.core.schema import ImageDocument


def test_class_name():
    """Test class name."""
    llm = EdenaiMultiModal()
    assert llm.class_name() == "edenai_multi_modal_llm"


def test_init():
    """Test initialization."""
    llm = EdenaiMultiModal(model_name="google/gemini-1.5-flash", api_key="fake-api-key")
    assert llm.model_name == "google/gemini-1.5-flash"
    assert llm.api_key == "fake-api-key"


def test_inheritance():
    """Test inheritance."""
    assert issubclass(EdenaiMultiModal, MultiModalLLM)

@patch("httpx.post")
def test_complete(mock_post):
    """Test completion."""
    mock_response = {
        'google/gemini-1.5-flash': {
            'generated_text': 'Yes, there is a lizard in the image.  It appears to be a type of agama lizard...',
            'messages': [
                {
                    'role': 'user',
                    'content': [{'type': 'text', 'content': {'text': 'is there a lizard in the image?'}}]
                },
                {
                    'role': 'assistant',
                    'content': [{'type': 'text', 'content': {'text': 'Yes, there is a lizard in the image...'}}]
                }
            ],
            'status': 'success',
        }
    }
    mock_post.return_value = MagicMock(status_code=200, json=lambda: mock_response)

    llm = EdenaiMultiModal(model_name="google/gemini-1.5-flash", api_key="fake-api-key")
    image_doc = ImageDocument(image="base64_encoded_string")

    response = llm.complete(prompt="is there a lizard in the image?", image_documents=[image_doc])

    assert response.text == "Yes, there is a lizard in the image.  It appears to be a type of agama lizard..."
    assert "prompt_tokens" in response.additional_kwargs
    assert "completion_tokens" in response.additional_kwargs
    assert "total_tokens" in response.additional_kwargs
    mock_post.assert_called_once()
