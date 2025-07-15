# import os
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# from judgeval.tracer import Tracer, wrap
# from judgeval.data import Example
# from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer
# from openai import OpenAI

# # Access env vars after loading
# client = wrap(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
# judgment = Tracer(project_name="faq_agent_demo")

import os
from dotenv import load_dotenv
from judgeval.tracer import Tracer, wrap
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer
from openai import OpenAI

load_dotenv()
client = wrap(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
judgment = Tracer(api_key=os.getenv("JUDGMENT_API_KEY"), organization_id=os.getenv("JUDGMENT_ORG_ID"), project_name="faq_agent_demo")

with open("faq_context.txt", "r") as f:
    faq_context = f.read()

@judgment.observe(span_type="function")
def answer_faq(question: str) -> str:
    prompt = f"""You are a helpful assistant answering customer FAQs.

Context:
{faq_context}

Question:
{question}

Only answer using the context above. If unsure, say 'I'm not sure.'"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

@judgment.observe(span_type="eval")
def evaluate_response(question: str, response: str):
    example = Example(
        input=question,
        actual_output=response,
        retrieval_context=[faq_context]
    )
    judgment.async_evaluate(
        scorers=[FaithfulnessScorer(threshold=0.5), AnswerRelevancyScorer(threshold=0.5)],
        example=example,
        model="gpt-3.5-turbo"
    )

if __name__ == "__main__":
    question = input("Ask a question: ")
    answer = answer_faq(question)
    print("\nAnswer:\n", answer)
    evaluate_response(question, answer)

