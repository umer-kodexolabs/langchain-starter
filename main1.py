import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

model = OllamaLLM(model="llama3.2:latest", streaming=True)

template = """
you are a chatbot assistant and would answer the user query
Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True:
    print("\n\n ----------------------------------")
    question = input("Ask a question to llm (q to quit): \n")
    print("\n\n")
    if question.lower() == "q":
        break

    for chunk in chain.stream({"question": question}):
        print(chunk, end="")
    print("\n\n ----------------------------------")


dict1 = {
    "emp1": {
        "name": "Lisa",
        "designation": "programmer",
        "age": "34",
        "salary": "54000",
    },
    "emp2": {"name": "Elis", "designation": "Trainee", "age": "24", "salary": "40000"},
}
out_file = open("myfile.json", "w")
json_str = json.dumps(dict1, indent=2)
print("json_str: ", json_str)

out_file.close()
