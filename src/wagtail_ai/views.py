import os

import anthropic
import openai
import tiktoken
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from every_ai import AIBackend
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .ai import get_ai_backend
from .prompts import Prompt, get_prompt_by_id

DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_MAX_TOKENS = 4096


class AIHandlerException(Exception):
    pass


def _splitter_length(string):
    """Return the number of tokens in a string, used by the Langchain
    splitter so we split based on tokens rather than characters."""
    encoding = tiktoken.encoding_for_model(DEFAULT_MODEL)
    return len(encoding.encode(string))


def _process_backend_request(full_prompt: str, backend: AIBackend[any]):
    """
    Method for processing prompt requests and handling errors.

    Errors will either be an API or Python library error, this method uses exception
    chaining to retain the original error and raise a more generic error message to be sent to the front-end.

    :param full_prompt: The full prompt to be sent to the AI backend.
    :param backend: The AI backend instance.

    :return: The response message from the AI backend.
    :raises AIHandlerException: Raised for specific error retaining the error scenarios to be communicated to the front-end.
    """
    try:
        message = backend.chat(user_messages=[full_prompt])
    except (anthropic.RateLimitError, openai.RateLimitError) as rate_limit_error:
        # Handle rate error limits (429 responses) separately to inform users.
        raise AIHandlerException(
            "Rate limit exceeded. Please try again later."
        ) from rate_limit_error
    except Exception as e:
        # Raise a more generic error to send to the front-end
        raise AIHandlerException(
            "Error processing request, Please try again later."
        ) from e

    return message


def _replace_handler(prompt: Prompt, text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_MAX_TOKENS, length_function=_splitter_length
    )
    texts = splitter.split_text(text)

    for split in texts:
        full_prompt = "\n".join([prompt.prompt, split])
        backend = get_ai_backend()
        message = _process_backend_request(full_prompt, backend)
        # Remove extra blank lines returned by the API
        message = os.linesep.join([s for s in message.splitlines() if s])
        text = text.replace(split, message)

    return text


def _append_handler(prompt: Prompt, text: str):
    tokens = _splitter_length(text)
    if tokens > DEFAULT_MAX_TOKENS:
        raise AIHandlerException("Cannot run completion on text this long")
    full_prompt = "\n".join([prompt.prompt, text])
    backend = get_ai_backend()
    message = _process_backend_request(full_prompt, backend)
    # Remove extra blank lines returned by the API
    message = os.linesep.join([s for s in message.splitlines() if s])

    return message


@csrf_exempt
def process(request):
    text = request.POST.get("text")
    prompt_idx = request.POST.get("prompt")
    prompt = get_prompt_by_id(int(prompt_idx))

    if not text:
        return JsonResponse(
            {
                "error": "No text provided - please enter some text before using AI \
                    features"
            },
            status=400,
        )

    if not prompt:
        return JsonResponse({"error": "Invalid prompt provided"}, status=400)

    handlers = {
        Prompt.Method.REPLACE: _replace_handler,
        Prompt.Method.APPEND: _append_handler,
    }

    handler = handlers[prompt.method]
    try:
        response = handler(prompt, text)
    except AIHandlerException as e:
        return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"message": response})
