import os

from haystack import Document, Pipeline
from haystack.components.rankers import (
    DiversityRanker,
    LostInTheMiddleRanker,
    TransformersSimilarityRanker,
)
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

similarity_ranker = TransformersSimilarityRanker(model_name_or_path="BAAI/bge-reranker-large", device="cuda", top_k=10)
diversity_ranker = DiversityRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-12-v2", device="cuda", top_k=10)
litm_ranker = LostInTheMiddleRanker(top_k=10)

dense_pipeline = Pipeline()
dense_pipeline.add_component(
    instance=text_embedder,
    name="text_embedder",
)
dense_pipeline.add_component(instance=dense_retriever, name="embedding_retriever")
dense_pipeline.add_component(instance=similarity_ranker, name="similarity_ranker")
dense_pipeline.add_component(instance=diversity_ranker, name="diversity_ranker")
dense_pipeline.add_component(instance=litm_ranker, name="litm_ranker")

dense_pipeline.connect("text_embedder", "embedding_retriever")
dense_pipeline.connect("embedding_retriever.documents", "similarity_ranker.documents")
dense_pipeline.connect("similarity_ranker.documents", "diversity_ranker.documents")
dense_pipeline.connect("diversity_ranker.documents", "litm_ranker.documents")

result_qrels_all = {}

for query_id, query in tqdm(queries.items()):
    output = dense_pipeline.run(
        {
            "text_embedder": {"text": query},
            "similarity_ranker": {"query": query},
            "diversity_ranker": {"query": query},
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