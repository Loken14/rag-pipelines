import os

from haystack import Document, Pipeline
from haystack.components.rankers import (
    LostInTheMiddleRanker,
    TransformersSimilarityRanker,
)
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.routers import DocumentJoiner
from haystack.document_stores import InMemoryDocumentStore
from instructor_embedders import InstructorTextEmbedder
from pinecone_haystack import PineconeDocumentStore
from pinecone_haystack.dense_retriever import PineconeDenseRetriever
from tqdm import tqdm

from rag_pipelines import BeirDataloader, BeirEvaluator

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

similarity_ranker = TransformersSimilarityRanker(model_name_or_path="BAAI/bge-reranker-large", device="cuda", top_k=10)
litm_ranker = LostInTheMiddleRanker(top_k=10)

hybrid_pipeline = Pipeline()

hybrid_pipeline.add_component(instance=sparse_retriever, name="bm25_retriever")
hybrid_pipeline.add_component(
    instance=text_embedder,
    name="text_embedder",
)
hybrid_pipeline.add_component(instance=dense_retriever, name="embedding_retriever")
hybrid_pipeline.add_component(instance=joiner, name="joiner")
hybrid_pipeline.add_component(instance=similarity_ranker, name="similarity_ranker")
hybrid_pipeline.add_component(instance=litm_ranker, name="litm_ranker")


hybrid_pipeline.connect("bm25_retriever", "joiner")
hybrid_pipeline.connect("text_embedder", "embedding_retriever")
hybrid_pipeline.connect("embedding_retriever", "joiner")
hybrid_pipeline.connect("joiner.documents", "similarity_ranker.documents")
hybrid_pipeline.connect("similarity_ranker.documents", "litm_ranker.documents")


result_qrels_all = {}

for query_id, query in tqdm(queries.items()):
    output = hybrid_pipeline.run(
        {
            "bm25_retriever": {"query": query},
            "text_embedder": {"text": query},
            "similarity_ranker": {"query": query},
            "litm_ranker": {"query": query},
        }
    )
    output_docs = output["litm_ranker"]["documents"]
    doc_qrels = {}
    for doc in output_docs:
        doc_qrels[doc.meta["corpus_id"]] = doc.score
    result_qrels_all[query_id] = doc_qrels

evaluator = BeirEvaluator(qrels, result_qrels_all, [3, 5, 7, 10])
ndcg, _map, recall, precision = evaluator.evaluate()
