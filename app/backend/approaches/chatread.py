from collections.abc import Awaitable
from typing import Any, Optional, Union, cast
from io import BytesIO

from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorQuery
from azure.storage.filedatalake.aio import FileSystemClient

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
from utils import extract_matching_filename
from prepdocslib.listfilestrategy import File

def get_file_name(user_query, all_paths):
    files = []
    try:
        for path in all_paths:
            files.append(path.name.split("/", 1)[1])
    except Exception as error:
        print("Error listing uploaded files", error)
    file_mentioned_in_query = extract_matching_filename(user_query, files)
    return file_mentioned_in_query


class ChatReadApproach(ChatApproach):
    """
    A multi-step approach that first uses OpenAI to turn the user's question into a search query,
    then parse relevant document, and then sends the conversation history,
    original user question, and search results to OpenAI to generate a response.
    """

    def __init__(
        self,
        *,
        search_client: SearchClient,
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
        self.search_client = search_client
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
        use_query_rewriting = True if overrides.get("query_rewriting") else False
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        original_user_query = messages[-1]["content"]
        parser = overrides.get("doc_parser")
        if not isinstance(original_user_query, str):
            raise ValueError("The most recent message content must be a string.")

        reasoning_model_support = self.GPT_REASONING_MODELS.get(self.chatgpt_model)
        if reasoning_model_support and (not reasoning_model_support.streaming and should_stream):
            raise Exception(
                f"{self.chatgpt_model} does not support streaming. Please use a different model or disable streaming."
            )

        query_messages = self.prompt_manager.render_prompt(
            self.query_rewrite_prompt, {"user_query": original_user_query, "past_messages": messages[:-1]}
        )
        tools: list[ChatCompletionToolParam] = self.query_rewrite_tools

        # STEP 1: Generate an optimized keyword search query based on the chat history and the last question

        chat_completion = cast(
            ChatCompletion,
            await self.create_chat_completion(
                self.chatgpt_deployment,
                self.chatgpt_model,
                messages=query_messages,
                overrides=overrides,
                response_token_limit=self.get_response_token_limit(
                    self.chatgpt_model, 100
                ),  # Setting too low risks malformed JSON, setting too high may affect performance
                temperature=0.0,  # Minimize creativity for search query generation
                tools=tools,
                reasoning_effort="low",  # Minimize reasoning for search query generation
            ),
        )

        query_text = self.get_search_query(chat_completion, original_user_query)

        # STEP 2: Parse the user uploaded document using file processor

        parser = overrides.get("doc_parser")
        user_blob_container_client = overrides.get("blob_client")
        user_oid = auth_claims["oid"]
        
        all_paths = user_blob_container_client.get_paths(path=user_oid)

        file_mentioned_by_client = get_file_name(original_user_query, all_paths)
        user_directory_client: FileSystemClient = user_blob_container_client.get_directory_client(user_oid)
        file_client = user_directory_client.get_file_client(file_mentioned_by_client)


        # Download the blob content to a buffer
        stream = BytesIO()
        file_client.download_blob().readinto(stream)
        
        # Reset the buffer's position to the beginning
        stream.seek(0)
        results = parser.fetch_file(File(content=stream, acls={"oids": [user_oid]}, url=file_client.url))

        # STEP 3: Generate a contextual and content specific answer using the search results and chat history
        
        text_sources = self.get_sources_content(results, use_semantic_captions, use_image_citation=False)
        messages = self.prompt_manager.render_prompt(
            self.answer_prompt,
            self.get_system_prompt_variables(overrides.get("prompt_template"))
            | {
                "include_follow_up_questions": bool(overrides.get("suggest_followup_questions")),
                "past_messages": messages[:-1],
                "user_query": original_user_query,
                "text_sources": text_sources,
            },
        )

        extra_info = ExtraInfo(
            DataPoints(text=text_sources),
            thoughts=[
                self.format_thought_step_for_chatcompletion(
                    title="Prompt to generate search query",
                    messages=query_messages,
                    overrides=overrides,
                    model=self.chatgpt_model,
                    deployment=self.chatgpt_deployment,
                    usage=chat_completion.usage,
                    reasoning_effort="low",
                ),
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
        return (extra_info, chat_coroutine)
