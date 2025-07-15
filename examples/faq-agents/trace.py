from judgeval.tracer import Tracer, wrap
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# client = wrap(OpenAI())  # tracks all LLM calls
# judgment = Tracer(project_name="my_project")
client = wrap(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"), organization_id=os.getenv("JUDGMENT_ORG_ID"), project_name="abc")


@judgment.observe(span_type="tool")
def format_question(question: str) -> str:
    # dummy tool
    return f"Question : {question}"

@judgment.observe(span_type="function")
# def run_agent(prompt: str) -> str:
#     task = format_question(prompt)
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": task}]
#     )
#     return response.choices[0].message.content
def run_agent(question: str):
    response = type("Response", (), {})()
    response.choices = [type("Choice", (), {"message": type("Msg", (), {"content": "Sorry, I can't answer that right now."})()})]
    return response

run_agent("What is the capital of the United States?")