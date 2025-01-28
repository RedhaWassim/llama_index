"""EdenAI API Utils."""

from http import HTTPStatus
from typing import Any, Dict, List, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
)
from llama_index.core.schema import ImageDocument


def get_usage_from_response(response: Dict) -> Dict:
    """
    Extract usage data from EdenAI response.
    If no usage data is available, returns default values.
    """
    original_response = response.get("original_response", {})
    usage_data = original_response.get("usage", {})
    
    return {
        "prompt_tokens": usage_data.get("prompt_tokens", 0),
        "completion_tokens": usage_data.get("completion_tokens", 0),
        "total_tokens": usage_data.get("total_tokens", 0),
    }

def edenai_response_to_completion_response(response: Any, model_name: str) -> CompletionResponse:
    """
    Convert EdenAI response to CompletionResponse.
    """
    try: 
        provider_response = response.get(model_name, {})
        content = provider_response.get("generated_text", "")

        usage_data = get_usage_from_response(provider_response)

        return CompletionResponse(text=content, raw=response, additional_kwargs=usage_data)
    except Exception:
        return CompletionResponse(text="", raw=response)


def edenai_response_to_chat_response(
    response: Any,
    model_name: str,
) -> ChatResponse:
    """
    Convert EdenAI response to ChatResponse.
    """
    try : 
        role = "assistant"  
        content = response[model_name]["generated_text"] or ""
        return ChatResponse(
            message=ChatMessage(role=role, content=content, raw=response),
        )
    except: 
        return ChatResponse(message=ChatMessage(), raw=response)


def chat_message_to_edenai_multi_modal_messages(
    chat_messages: Sequence[ChatMessage],
) -> List[Dict]:
    """
    Convert ChatMessage objects to EdenAI multimodal message format.
    """
    messages = []
    for msg in chat_messages:
        messages.append({"role": msg.role.value, "content": msg.content})
    return messages


def create_edenai_multi_modal_chat_message(
    prompt: str, role: str, image_documents: Sequence[ImageDocument]
) -> ChatMessage:
    """
    Create a ChatMessage object for multimodal requests, including text and images.
    """
    if not image_documents:
        message = ChatMessage(role=role, content=[{"type": "text", "content": {"text": prompt}}])
    else:
        content = []
        for image_document in image_documents:
            content.append(
                {
                    "type": "media_url",
                    "content": {
                        "media_url": (
                            image_document.image_url
                            if image_document.image_url is not None
                            else image_document.image_path
                        ),
                        "media_type": "image/jpeg",
                    },
                }
            )
        content.append({"type": "text", "content": {"text": prompt}})
        message = ChatMessage(role=role, content=content)

    return message


def load_local_images(local_images: List[str]) -> List[ImageDocument]:
    """
    Load images from local paths into ImageDocument objects.
    """
    image_documents = []
    for _, img in enumerate(local_images):
        new_image_document = ImageDocument(image_path=img)
        image_documents.append(new_image_document)
    return image_documents
