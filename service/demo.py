import gradio as gr
import requests

def answer(message):
    try:
        url = "http://localhost:8000/message"
        payload = {"message": message}
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "No response from server.")
    except Exception as e:
        return f"Error: {e}"

gr.ChatInterface(
    answer,
    type="messages",
    chatbot=gr.Chatbot(height=600),
    textbox=gr.Textbox(placeholder="무엇이든 물어보세요!", container=False, scale=7),
    title="Askku",
    description="무엇이든 물어보세요!",
    theme="ocean",
    examples=["다음학기 군휴학하는데 도전학기 들어도 돼?","복수전공 포기 언제할 수 있어?"],
    cache_examples=False,
).launch()
