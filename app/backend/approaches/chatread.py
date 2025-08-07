from collections.abc import Awaitable
from typing import Any, Optional, Union, cast
from io import BytesIO

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from azure.storage.filedatalake.aio import FileSystemClient
from azure.search.documents.models import QueryCaptionResult

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from approaches.approach import DataPoints, ExtraInfo, ThoughtStep
from approaches.chatapproach import ChatApproach
from approaches.promptmanager import PromptManager
from core.authentication import AuthenticationHelper
from utils import get_file_name, sections_to_documents
from prepdocslib.listfilestrategy import File
from prepdocslib.searchmanager import Section

import logging
logger = logging.getLogger("scripts")

class ChatReadApproach(ChatApproach):
    """
    A multi-step approach that first uses OpenAI to turn the user's question into a search query,
    then parse relevant document, and then sends the conversation history,
    original user question, and search results to OpenAI to generate a response.
    """

    def __init__(
        self,
        *,
        auth_helper: AuthenticationHelper,
        openai_client: AsyncOpenAI,
        chatgpt_model: str,
        chatgpt_deployment: Optional[str],  # Not needed for non-Azure OpenAI
        embedding_deployment: Optional[str],  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embedding_model: str,
        embedding_dimensions: int,
        sourcepage_field: str,
        content_field: str,
        query_language: str,
        query_speller: str,
        prompt_manager: PromptManager,
        reasoning_effort: Optional[str] = None,
    ):
        self.openai_client = openai_client
        self.auth_helper = auth_helper
        self.chatgpt_model = chatgpt_model
        self.chatgpt_deployment = chatgpt_deployment
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.query_language = query_language
        self.query_speller = query_speller
        self.prompt_manager = prompt_manager
        self.query_rewrite_prompt = self.prompt_manager.load_prompt("chat_query_rewrite.prompty")
        self.query_rewrite_tools = self.prompt_manager.load_tools("chat_query_rewrite_tools.json")
        self.answer_prompt = self.prompt_manager.load_prompt("chat_answer_question.prompty")
        self.reasoning_effort = reasoning_effort
        self.include_token_usage = True

    async def run_until_final_call(
        self,
        messages: list[ChatCompletionMessageParam],
        overrides: dict[str, Any],
        auth_claims: dict[str, Any],
        should_stream: bool = False,
    ) -> tuple[ExtraInfo, Union[Awaitable[ChatCompletion], Awaitable[AsyncStream[ChatCompletionChunk]]]]:

        original_user_query = messages[-1]["content"]
        if "confirm" in original_user_query.lower() or "submit" in original_user_query.lower():
            original_user_query = "Extract JSON"

        reasoning_model_support = self.GPT_REASONING_MODELS.get(self.chatgpt_model)
        if reasoning_model_support and (not reasoning_model_support.streaming and should_stream):
            raise Exception(
                f"{self.chatgpt_model} does not support streaming. Please use a different model or disable streaming."
            )

        documents = overrides.get("documents")
        if documents is None:
            text_sources = []
        else:
            text_sources = self.get_sources_content(documents, use_semantic_captions=False, use_image_citation=False)

        logger.info(f"File name: {overrides.get('file_name', '')}")

        messages = self.prompt_manager.render_prompt(
            self.answer_prompt,
            self.get_system_prompt_variables(overrides.get("prompt_template"))
            | {
                "include_follow_up_questions": bool(overrides.get("suggest_followup_questions")),
                "past_messages": messages[:-1],
                "user_query": original_user_query,
                "text_sources": text_sources,
                "file_name": overrides.get("file_name", ""),
            },
        )

        chat_coroutine = cast(
            Union[Awaitable[ChatCompletion], Awaitable[AsyncStream[ChatCompletionChunk]]],
            self.create_chat_completion(
                self.chatgpt_deployment,
                self.chatgpt_model,
                messages,
                overrides,
                self.get_response_token_limit(self.chatgpt_model, 1024),
                should_stream,
            ),
        )

        extra_info = ExtraInfo(
            DataPoints(text=text_sources),
            thoughts=[
                self.format_thought_step_for_chatcompletion(
                    title="Prompt to generate answer",
                    messages=messages,
                    overrides=overrides,
                    model=self.chatgpt_model,
                    deployment=self.chatgpt_deployment,
                    usage=None,
                ),
            ],
        )


        return (extra_info, chat_coroutine)
