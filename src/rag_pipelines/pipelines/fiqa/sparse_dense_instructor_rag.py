import os
from typing import Any, Dict

from haystack import Document, Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.routers import DocumentJoiner
from haystack.document_stores import InMemoryDocumentStore
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


sparse_document_store = InMemoryDocumentStore()
sparse_document_store.write_documents(documents_corp)
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

sparse_retriever = InMemoryBM25Retriever(document_store=sparse_document_store, top_k=10)

joiner = DocumentJoiner(join_mode="reciprocal_rank_fusion")


hybrid_pipeline = Pipeline()

hybrid_pipeline.add_component(instance=sparse_retriever, name="bm25_retriever")
hybrid_pipeline.add_component(
    instance=text_embedder,
    name="text_embedder",
)
hybrid_pipeline.add_component(instance=dense_retriever, name="embedding_retriever")
hybrid_pipeline.add_component(instance=joiner, name="joiner")

hybrid_pipeline.add_component(instance=PromptBuilder(template=prompt_template), name="prompt_builder")
hybrid_pipeline.add_component(
    instance=HuggingFaceLocalGenerator(
        model="google/flan-t5-small",
        task="text2text-generation",
        generation_kwargs={"max_new_tokens": 100, "temperature": 0.5, "do_sample": True},
    ),
    name="llm",
)
hybrid_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")


hybrid_pipeline.connect("bm25_retriever", "joiner")
hybrid_pipeline.connect("text_embedder", "embedding_retriever")
hybrid_pipeline.connect("embedding_retriever", "joiner")
hybrid_pipeline.connect("joiner.documents", "prompt_builder.documents")
hybrid_pipeline.connect("prompt_builder", "llm")
hybrid_pipeline.connect("llm.replies", "answer_builder.replies")
hybrid_pipeline.connect("joiner.documents", "answer_builder.documents")


answers: Dict[str, Any] = {}

for query_id, query in tqdm(queries.items()):
    output = hybrid_pipeline.run(
        {
            "bm25_retriever": {"query": query},
            "text_embedder": {"text": query},
            "prompt_builder": {"question": query},
            "answer_builder": {"query": query},
        }
    )
    generated_answer = output["answer_builder"]["answers"][0]
    answers[query_id] = generated_answer
