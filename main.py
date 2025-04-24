from transformers import pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate


# pipe = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")

hf = HuggingFacePipeline.from_model_id(
    model_id="mistralai/Mistral-7B-v0.1",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 2000},
)


template = """
you are a chatbot assistant and would answer the user query
Here is the question to answer: {question}

"""

prompt = ChatPromptTemplate(template)

chain = prompt | hf

while True:
    print("\n\n ----------------------------------")
    question = input("Ask a question to llm (q to quit)")
    print("\n\n")
    if question == "q":
        break

    result = chain.invoke({"question": "What is LangChain?"})
    print(result)
    print("\n\n ----------------------------------")
