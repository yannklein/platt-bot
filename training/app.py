"""
Platt Lorrain Chat - Gradio App for Hugging Face Spaces
"""

import os
import gradio as gr
from together import Together

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")

SYSTEM_PROMPT = """Du bist ein freundlicher Assistent, der Platt Lorrain (Francique rhénan lorrain) spricht.
Du antwortest IMMER auf Platt, egal in welcher Sprache die Frage gestellt wird.
Platt ist ein deutscher Dialekt aus Lothringen, nahe am Hochdeutschen aber mit eigenen Regeln.
Sei natürlich, freundlich und hilfsbereit - immer auf Platt!"""


def chat_stream(message: str, history: list):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for item in history:
        if isinstance(item, dict):
            messages.append({"role": item["role"], "content": item["content"]})
        else:
            user_msg, assistant_msg = item
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=256,
        temperature=0.7,
        stream=True,
    )

    partial_message = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            partial_message += chunk.choices[0].delta.content
            yield partial_message


EXAMPLES = [
    "Wie geht es dir?",
    "Was machst du heute?",
    "Erzähl mir etwas über Lothringen",
]

demo = gr.ChatInterface(
    fn=chat_stream,
    title="🗣️ Platt Lorrain Chat",
    description="Schwätz Platt mit mir! Ask in German or Platt, I'll respond in Platt!",
    examples=EXAMPLES,
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
