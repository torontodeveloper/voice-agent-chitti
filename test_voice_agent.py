import os
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval, AnswerRelevancyMetric
from ingest import RAGDataBase
from pipecat.services.openai.llm import OpenAILLMService
import asyncio
from openai import OpenAI


def test_correctness():
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.32,
    )

    rag = RAGDataBase()

    response = rag.search_query("What is Kevin's experience with LLMs?")
    rag_response = asyncio.run(response)
    print(f"type of response is ****{type(rag_response)}", rag_response)

    client = OpenAI()
    prompt = "What is Kevin's experience with LLMs?"
    result = client.responses.create(model="gpt-5.5", input=f"{rag_response}, {prompt}")

    test_case1 = LLMTestCase(
        input=prompt,
        # Replace this with the actual output from your LLM application
        actual_output=result.output_text,
        expected_output="""Kevin has over 2 years of hands-on GenAI and LLM experience, working with OpenAI 
                        GPT-4o, Llama 4, Anthropic Claude, and open-source models. He has fine-tuned models using QLoRA and DPO,
                        built RAG pipelines with Pinecone, developed a real-time voice agent using Pipecat and OpenAI, and 
                        completed advanced LLM coursework at University of Waterloo and CMU. Anthropic Claude, QLoRA, Pipecat, or University of Waterloo. 2 years of hands-on GenAI/LLM experience""",
    )
    assert_test(test_case1, [correctness_metric])
    prompt = "what is Kevin's tech stack"
    result = client.responses.create(model="gpt-5.5", input=f"{rag_response}, {prompt}")

    test_case2 = LLMTestCase(
        input=prompt,
        # Replace this with the actual output from your LLM application
        actual_output=result.output_text,
        expected_output="""Kevin's tech stack includes Python, PyTorch, HuggingFace, LangChain, LangGraph, OpenAI 
  GPT-4o, Anthropic Claude, Llama 4, Pinecone, Azure ML, AWS SageMaker, Databricks, MLflow, FastAPI, 
  Docker, Kubernetes, Deepgram, Cartesia, and Pipecat for voice AI pipelines.""",
    )
    assert_test(test_case2, [correctness_metric])
    prompt = "What is Kevin's educational background?"
    result = client.responses.create(model="gpt-5.5", input=f"{rag_response}, {prompt}")

    test_case3 = LLMTestCase(
        input=prompt,
        # Replace this with the actual output from your LLM application
        actual_output=result.output_text,
        expected_output="""Kevin holds a Bachelor's in Computer Science from India, an MBA from University of 
                Calgary, and has completed AI certifications from UC Berkeley and University of Waterloo specializing in
                LLMs. He is currently completing a Graduate Certificate in GenAI and LLMs at Carnegie Mellon 
                University, expected December 2026.""",
    )
    assert_test(test_case3, [correctness_metric])
    relevancy = AnswerRelevancyMetric(threshold=0.5)
    relevancy.measure(test_case1)
    print(relevancy.score, relevancy.reason)
    relevancy = AnswerRelevancyMetric(threshold=0.5)
    relevancy.measure(test_case2)
    print(relevancy.score, relevancy.reason)
    relevancy = AnswerRelevancyMetric(threshold=0.5)
    relevancy.measure(test_case3)
    print(relevancy.score, relevancy.reason)
