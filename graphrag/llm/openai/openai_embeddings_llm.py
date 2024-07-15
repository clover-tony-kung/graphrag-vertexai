# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The EmbeddingsLLM class."""

from langchain_google_vertexai import VertexAIEmbeddings
from typing_extensions import Unpack

from graphrag.llm.base import BaseLLM
from graphrag.llm.types import EmbeddingInput, EmbeddingOutput, LLMInput

from .openai_configuration import OpenAIConfiguration
from .types import OpenAIClientTypes


class OpenAIEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    """A text-embedding generator LLM."""

    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        self.client = client
        self.configuration = configuration

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        embedding_model = VertexAIEmbeddings(
            model_name="text-embedding-004", temperature=0.0
        )
        return embedding_model.embed(input)
