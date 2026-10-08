import os

from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()


API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


model = None

if API_KEY:
    genai.configure(
        api_key=API_KEY
    )

    model = genai.GenerativeModel(
        "gemini-flash-latest"
    )


def generate_answer(question, context):

    if model is None:
        raise ValueError(
            "GEMINI_API_KEY is missing from .env"
        )

    prompt = f"""
You are an HR Assistant.

Answer the employee's question using ONLY
the information provided in the context.

If the answer cannot be found in the context,
say:

"I couldn't find this information in the company documents."

Do not make up information.

Keep the answer clear, concise and professional.

Context:
----------------
{context}
----------------

Employee Question:
{question}

Answer:
"""

    response = model.generate_content(
        prompt
    )

    return response.text
