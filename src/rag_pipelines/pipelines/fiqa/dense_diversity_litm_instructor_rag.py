import os
from typing import Any, Dict

from haystack import Document, Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.rankers import (
    DiversityRanker,
    LostInTheMiddleRanker,
)
from instructor_embedders import InstructorTextEmbedder
from pinecone_haystack import PineconeDocumentStore
from pinecone_haystack.dense_retriever import PineconeDenseRetriever
from tqdm import tqdm

from rag_pipelines import BeirDataloader

prompt_template = """
    Given these documents, answer the question.\nDocuments:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{question}}
    \nAnswer:"""

dataset = "fiqa"
data_loader = BeirDataloader(dataset)
data_loader.download_and_unzip()
corpus, queries, qrels = data_loader.load()

documents_corp = [
    Document(
        content=text_dict["text"],
        meta={"corpus_id": str(corpus_id), "title": text_dict["title"]},
    )
    for corpus_id, text_dict in corpus.items()
]

dense_document_store = PineconeDocumentStore(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="gcp-starter",
    index="fiqa",
    namespace="default",
    dimension=768,
)

dense_retriever = PineconeDenseRetriever(document_store=dense_document_store, top_k=10)
query_instruction = "Represent the financial question for retrieving supporting documents:"
text_embedder = InstructorTextEmbedder(
    model_name_or_path="hkunlp/instructor-xl",
    instruction=query_instruction,
    device="cuda",
)

diversity_ranker = DiversityRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2", device="cuda", top_k=10)
litm_ranker = LostInTheMiddleRanker(top_k=10)

dense_pipeline = Pipeline()
dense_pipeline.add_component(
    instance=text_embedder,
    name="text_embedder",
)
dense_pipeline.add_component(instance=dense_retriever, name="embedding_retriever")
dense_pipeline.add_component(instance=diversity_ranker, name="diversity_ranker")
dense_pipeline.add_component(instance=litm_ranker, name="litm_ranker")

dense_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
dense_pipeline.add_component(
    instance=HuggingFaceLocalGenerator(
        model="google/flan-t5-small",
        task="text2text-generation",
        generation_kwargs={"max_new_tokens": 100, "temperature": 0.5, "do_sample": True},
    ),
    name="llm",
)
dense_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")

dense_pipeline.connect("text_embedder", "embedding_retriever")
dense_pipeline.connect("embedding_retriever.documents", "diversity_ranker.documents")
dense_pipeline.connect("diversity_ranker.documents", "litm_ranker.documents")
dense_pipeline.connect("litm_ranker.documents", "prompt_builder.documents")
dense_pipeline.connect("prompt_builder", "llm")
dense_pipeline.connect("llm.replies", "answer_builder.replies")
dense_pipeline.connect("litm_ranker.documents", "answer_builder.documents")

answers: Dict[str, Any] = {}

for query_id, query in tqdm(queries.items()):
    output = dense_pipeline.run(
        {
            "text_embedder": {"text": query},
            "diversity_ranker": {"query": query},
            "litm_ranker": {"query": query},
            "prompt_builder": {"question": query},
            "answer_builder": {"query": query},
        }
    )
    generated_answer = output["answer_builder"]["answers"][0]
    answers[query_id] = generated_answer
