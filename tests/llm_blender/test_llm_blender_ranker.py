import pytest
from haystack import ComponentError, Pipeline
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.dataclasses import Document, GeneratedAnswer
from haystack.document_stores.in_memory import InMemoryDocumentStore

from rag_pipelines.llm_blender import LLMBlenderRanker


class TestLLMBlenderRanker:
    def test_init(self):
        """
        Test that the LLMBlenderRanker is initialized correctly with default parameters.
        """
        llm_ranker = LLMBlenderRanker(model="llm-blender/PairRM")

        assert llm_ranker.model_name_or_path == "llm-blender/PairRM"
        assert llm_ranker.device == "cpu"
        assert llm_ranker.model_kwargs == {}

    def test_init_custom_parameters(self):
        """
        Test that the LLMBlenderRanker is initialized correctly with custom parameters.
        """
        llm_ranker = LLMBlenderRanker(model="llm-blender/PairRM", device="cuda", model_kwargs={"cache_dir": "/models"})

        assert llm_ranker.model_name_or_path == "llm-blender/PairRM"
        assert llm_ranker.device == "cuda"
        assert llm_ranker.model_kwargs == {"cache_dir": "/models"}

    def test_run_without_warm_up(self):
        """
        Test that ranker loads the PairRanker model correctly during warm up.
        """
        llm_ranker = LLMBlenderRanker(model="llm-blender/PairRM")
        answers = [
            GeneratedAnswer(data="answer 1", query="query 1", documents=[]),
            GeneratedAnswer(data="answer 2", query="query 2", documents=[]),
        ]

        assert llm_ranker.model is None
        with pytest.raises(ComponentError, match="The component LLMBlenderRanker wasn't warmed up."):
            llm_ranker.run(answers=[answers])

        llm_ranker.warm_up()
        assert llm_ranker.model is not None

    def test_generation_of_inputs_and_candidates(self):
        """
        Test that the LLMBlenderRanker generates the correct inputs and candidates for a list of answers.
        """
        llm_ranker = LLMBlenderRanker(model="llm-blender/PairRM")
        answers = [
            GeneratedAnswer(data="answer 1", query="query 1", documents=[]),
            GeneratedAnswer(data="answer 2", query="query 2", documents=[]),
        ]
        inputs, candidates, meta = llm_ranker._generate_inputs_candidates([answers])

        assert inputs == ["query 1", "query 2"]
        assert candidates == [["answer 1"], ["answer 2"]]
        assert meta == [[{}], [{}]]

    def test_generation_of_inputs_and_candidates_for_same_input(self):
        """
        Test that the LLMBlenderRanker generates the correct inputs and candidates for a list of answers with the same
        input.
        """
        llm_ranker = LLMBlenderRanker(model="llm-blender/PairRM")
        answers_1 = [
            GeneratedAnswer(data="answer 1", query="query 1", documents=[]),
            GeneratedAnswer(data="answer 2", query="query 2", documents=[]),
        ]
        answers_2 = [
            GeneratedAnswer(data="answer 3", query="query 1", documents=[]),
            GeneratedAnswer(data="answer 4", query="query 2", documents=[]),
        ]

        inputs, candidates, meta = llm_ranker._generate_inputs_candidates([answers_1, answers_2])

        assert inputs == ["query 1", "query 2"]
        assert candidates == [["answer 1", "answer 3"], ["answer 2", "answer 4"]]
        assert meta == [[{}, {}], [{}, {}]]

    def test_ranking_candidates(self):
        """
        Test that the LLMBlenderRanker ranks the candidates correctly for a list of inputs and candidates.
        """
        llm_ranker = LLMBlenderRanker(model="llm-blender/PairRM")
        inputs = ["query 1", "query 2"]
        candidates = [["answer 1", "answer 2"], ["answer 3", "answer 4"]]
        ranks = [[1, 0], [0, 1]]
        meta = [[{"answer": "answer 1"}, {"answer": "answer 2"}], [{"answer": "answer 3"}, {"answer": "answer 4"}]]
        ranked_answers = llm_ranker._generate_answers_ranked_candidates(inputs, candidates, ranks, meta)

        assert ranked_answers == [
            GeneratedAnswer(data="answer 2", query="query 1", documents=[], meta={"answer": "answer 2"}),
            GeneratedAnswer(data="answer 1", query="query 1", documents=[], meta={"answer": "answer 1"}),
            GeneratedAnswer(data="answer 3", query="query 2", documents=[], meta={"answer": "answer 3"}),
            GeneratedAnswer(data="answer 4", query="query 2", documents=[], meta={"answer": "answer 4"}),
        ]

    def test_run(self):
        """
        Test that the LLMBlenderRanker ranks the answers correctly.
        """
        llm_ranker = LLMBlenderRanker(model="llm-blender/PairRM")
        llm_ranker.warm_up()

        answers_1 = [
            GeneratedAnswer(data="answer 1", query="query 1", documents=[]),
            GeneratedAnswer(data="answer 2", query="query 2", documents=[]),
        ]
        answers_2 = [
            GeneratedAnswer(data="answer 3", query="query 1", documents=[]),
            GeneratedAnswer(data="answer 4", query="query 2", documents=[]),
        ]

        output = llm_ranker.run(answers=[answers_1, answers_2])
        ranked_answers = output["answers"]

        assert ranked_answers == [
            GeneratedAnswer(data="answer 3", query="query 1", documents=[], meta={}),
            GeneratedAnswer(data="answer 1", query="query 1", documents=[], meta={}),
            GeneratedAnswer(data="answer 4", query="query 2", documents=[], meta={}),
            GeneratedAnswer(data="answer 2", query="query 2", documents=[], meta={}),
        ]

    def test_run_empty_answers(self):
        """
        Test that the LLMBlenderRanker handles an empty list of answers correctly.
        """
        llm_ranker = LLMBlenderRanker(model="llm-blender/PairRM")
        llm_ranker.warm_up()

        output = llm_ranker.run(answers=[])
        ranked_answers = output["answers"]

        assert ranked_answers == []

    def test_rag_pipeline(self):
        """
        Test that the LLMBlenderRanker can be used in a RAG pipeline.
        """
        prompt_template = """
        Given these documents, answer the question.\nDocuments:
        {% for doc in documents %}
            {{ doc.content }}
        {% endfor %}

        \nQuestion: {{question}}
        \nAnswer:
        """

        document_store = InMemoryDocumentStore()
        bm25_retriever = InMemoryBM25Retriever(document_store=document_store)

        prompt_builder = PromptBuilder(template=prompt_template)
        generator = HuggingFaceLocalGenerator(
            model="google/flan-t5-small",
            task="text2text-generation",
            generation_kwargs={"max_new_tokens": 20, "temperature": 0.1, "do_sample": True, "num_return_sequences": 3},
        )

        answer_builder = AnswerBuilder()

        document_store_2 = InMemoryDocumentStore()
        bm25_retriever_2 = InMemoryBM25Retriever(document_store=document_store_2)

        prompt_builder_2 = PromptBuilder(template=prompt_template)
        generator_2 = HuggingFaceLocalGenerator(
            model="google/flan-t5-small",
            task="text2text-generation",
            generation_kwargs={"max_new_tokens": 20, "temperature": 0.1, "do_sample": True, "num_return_sequences": 3},
        )

        answer_builder_2 = AnswerBuilder()

        llm_ranker = LLMBlenderRanker(model="llm-blender/PairRM")

        rag_pipeline = Pipeline()
        rag_pipeline.add_component(instance=bm25_retriever, name="retriever")
        rag_pipeline.add_component(instance=prompt_builder, name="prompt_builder")
        rag_pipeline.add_component(
            instance=generator,
            name="llm",
        )
        rag_pipeline.add_component(instance=answer_builder, name="answer_builder")

        rag_pipeline.add_component(instance=bm25_retriever_2, name="retriever_2")
        rag_pipeline.add_component(instance=prompt_builder_2, name="prompt_builder_2")
        rag_pipeline.add_component(
            instance=generator_2,
            name="llm_2",
        )
        rag_pipeline.add_component(instance=answer_builder_2, name="answer_builder_2")
        rag_pipeline.add_component(instance=llm_ranker, name="llm_ranker")

        # Connect the first pipeline
        rag_pipeline.connect("retriever", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder", "llm")
        rag_pipeline.connect("llm.replies", "answer_builder.replies")
        rag_pipeline.connect("retriever", "answer_builder.documents")
        rag_pipeline.connect("answer_builder", "llm_ranker")

        # Connect the second pipeline
        rag_pipeline.connect("retriever_2", "prompt_builder_2.documents")
        rag_pipeline.connect("prompt_builder_2", "llm_2")
        rag_pipeline.connect("llm_2.replies", "answer_builder_2.replies")
        rag_pipeline.connect("retriever_2", "answer_builder_2.documents")
        rag_pipeline.connect("answer_builder_2", "llm_ranker")

        # Populate the first document store
        documents = [
            Document(content="My name is Jean and I live in Paris."),
            Document(content="My name is Mark and I live in Berlin."),
            Document(content="My name is Giorgio and I live in Rome."),
        ]
        rag_pipeline.get_component("retriever").document_store.write_documents(documents)

        # Populate the second document store
        documents = [
            Document(content="My name is Giorgio and I live in Paris."),
            Document(content="My name is Mark and I live in Berlin."),
            Document(content="My name is Jean and I live in Rome."),
        ]
        rag_pipeline.get_component("retriever_2").document_store.write_documents(documents)

        question = "Who lives in Paris?"
        outputs = rag_pipeline.run(
            {
                "retriever": {"query": question},
                "prompt_builder": {"question": question},
                "answer_builder": {"query": question},
                "retriever_2": {"query": question},
                "prompt_builder_2": {"question": question},
                "answer_builder_2": {"query": question},
            }
        )

        answers = outputs["llm_ranker"]["answers"]

        assert answers == [
            GeneratedAnswer(data="Jean", query="Who lives in Paris?", documents=[], meta={}),
            GeneratedAnswer(data="Jean", query="Who lives in Paris?", documents=[], meta={}),
            GeneratedAnswer(data="Jean", query="Who lives in Paris?", documents=[], meta={}),
            GeneratedAnswer(data="Giorgio", query="Who lives in Paris?", documents=[], meta={}),
            GeneratedAnswer(data="Giorgio", query="Who lives in Paris?", documents=[], meta={}),
            GeneratedAnswer(data="Giorgio", query="Who lives in Paris?", documents=[], meta={}),
        ]
