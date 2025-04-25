from langchain_ollama.llms import OllamaLLM
from langchain.tools import StructuredTool
from pydantic import BaseModel
from typing import List

# Setup LLM
llm = OllamaLLM(model="llama3.2:latest", streaming=True)


class FollowupInput(BaseModel):
    question: str
    answer: str


class SummaryInput(BaseModel):
    qna_list: List[FollowupInput]


# 2. Predefined Questions (20 questions)
predefined_questions = [
    "What are your hobbies or interests?",
    "Do you have any pets?",
    "What type of music do you enjoy listening to?",
    "What’s your daily routine like?",
    "Do you enjoy traveling?",
    "What’s the most important thing in your life right now?",
    "How do you like to spend your weekends?",
    "Do you prefer spending time indoors or outdoors?",
    "Are you currently working on any personal projects or goals?",
    "What’s one thing you would like to learn or experience in the future?",
    # "What's your project about?",
    # "Who is your target audience?",
    # "What problem does your project solve?",
    # "What technologies are you using?",
    # "How will users interact with it?",
    # "What platforms will it support?",
    # "What is your monetization strategy?",
    # "How do you plan to market it?",
    # "What is your timeline?",
    # "What challenges do you expect?",
    # "How do you collect user feedback?",
    # "How do you ensure user privacy?",
    # "What's your team structure?",
    # "What are your long-term goals?",
    # "Who are your competitors?",
    # "How do you differentiate yourself?",
    # "How will you handle scalability?",
    # "Do you plan to raise funding?",
    # "What KPIs will you track?",
    # "What is your backup plan?"
]


# Tool: Followup Generator
def generate_followups(question, answer):
    prompt = f"""
You are a helpful assistant. Based on the following question and answer, generate 3 follow-up questions.

Question: {question}
Answer: {answer}

Only return the questions numbered as 1., 2., 3.
"""
    result = llm.invoke(prompt)
    return result


followup_tool = StructuredTool.from_function(
    name="FollowupGenerator",
    description="Generates 3 follow-up questions based on the main question and answer.",
    func=generate_followups,
    args_schema=FollowupInput,
)


# Tool: Summary Generator
def generate_summary(qna_list):
    format = "\n".join([f"Q: {item.question}\n A: {item.answer}" for item in qna_list])
    prompt = f"""
    Given the following Q&A conversation, generate a detailed summary.
    {format}
    Summary:
    """
    return llm.invoke(prompt)


summary_tool = StructuredTool.from_function(
    name="SummaryGenerator",
    description="Generates summary based on all user answers.",
    func=generate_summary,
    args_schema=SummaryInput,
)


def main():
    qna_list = []

    for index, q in enumerate(predefined_questions, start=1):
        print(f"\n Q{index}: {q}")
        answer = input("Your answer: ")
        qna_list.append({"question": q, "answer": answer})

        follow_up = (
            input("Would you like follow-up questions? (yes/no): ").strip().lower()
        )
        if follow_up == "yes":
            followups_questions = followup_tool.invoke(
                {"question": q, "answer": answer}
            )

            for line in followups_questions.strip().split("\n"):

                if line.strip() == "":
                    continue

                print(f"\n{line}")

                follow_up_answer = input("Your answer: ")
                qna_list.append({"question": line, "answer": follow_up_answer})

    print("\nGenerating summary...")
    qna_data = {"qna_list": qna_list}
    summary = summary_tool.invoke(qna_data)
    print("\n--- Summary ---")
    print(summary)


if __name__ == "__main__":
    main()
