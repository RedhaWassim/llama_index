# LlamaIndex Multi Modal Llms Integration: EdenAI

# EdenAI Platform
EdenAI is a platform that aggregates and provides access to multiple Large Language Models (LLMs) under a single account. With EdenAI, users can interact with different LLMs from different providers using a unified API.

more informations : https://edenai.co/

## Installation

```bash
pip install llama-index-multi-modal-llms-edenai
```

## Usage

Here's how to use the EdenAI multi-modal integration:

### Basic Usage

```python
from llama_index.multi_modal_llms.edenai import EdenaiMultiModal
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import ImageDocument
import os

# Initialize the model (credentials can be provided through environment variables)
# get api key from env
api_key = os.getenv("EDENAI_API_KEY")
llm = EdenaiMultiModal(
    model="openai/gpt-4o", 
    # or other for more info check the documentation : https://docs.edenai.co/reference/multimodal_multimodal_chat_create
    temperature=0.0,
    max_tokens=300,
    api_key=api_key,
)

# Method 1: Load images using SimpleDirectoryReader
image_documents = SimpleDirectoryReader(
    input_files=["path/to/image.jpg"]
).load_data()

# Method 2: Create image documents directly
image_doc = ImageDocument(
    image_path="/path/to/image.jpg",  # Local file path
    # OR
    image="base64_encoded_image_string",  # Base64 encoded image
)

# Get a completion with both text and image
response = llm.complete(
    prompt="Describe this image in detail:",
    image_documents=image_documents,  # or [image_doc]
)

print(response.text)
```


### Supported Models

Currently supported multi-modal models in EdenAI:

- `anthropic/claude-3-5-sonnet-latest`
- `anthropic/claude-3-5-haiku-latest`
- `anthropic/claude-3-opus-latest`
- `google/gemini-1.5-flash`
- `google/gemini-1.5-pro`
- `openai/o1`
- `openai/gpt-4`
- `openai/gpt-4o`
- `openai/gpt-4-turbo`
- `openai/gpt-4o-mini`
- `openai/o1-mini`


