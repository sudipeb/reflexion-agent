import datetime
import os
from dotenv import load_dotenv

from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set. Add it to your .env file.")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key,
    )


actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time:{time}
1. {first_instruction}
2. Reflect and Critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the users question above using the required format"),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)
revise_instructions = """Revise your previous answer using the new information.
-You should use the previous critique to add  important information to your answer
-Add a "References" section to the bottom of your answer(which does not count towards the word limit).In the form of: 
    -[1] https://example.com
    -[2] https://example.com
-You shouls use the previous critique  to remove superfluous  information from your answer and make SURE it is not more than 200 words.
"""
revisor_prompt_template = actor_prompt_template.partial(first_instruction=revise_instructions)


llm = get_llm()

# Keep these module-level runnables for the LangGraph nodes in main.py.
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)
revisor = revisor_prompt_template | llm.bind_tools(
    tools=[ReviseAnswer], tool_choice="ReviseAnswer"
)


if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-powered SOC/ autonomous soc problem domain,"
        "list startups that do that and raised captial."
    )
    res = first_responder.invoke(input={"messages": [human_message]})
    print(res)
