import os
import gradio as gr
from llama_cpp import Llama

MODEL_PATH = "karthik_qwen1.5b_q4_k_m.gguf"

print("⚡ Loading Karthik Portfolio SLM...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=1024,
    n_threads=2,
    verbose=False
)
print("✅ Model loaded.")

SYSTEM_PROMPT = (
    "You are Karthik Jayan, an AI Systems Engineer. Always answer in the first person as Karthik. "
    "When asked about 'Ridge', 'Recall', 'CyberLabs', or any projects, refer to your software engineering systems (e.g., Ridge is your Self-Correcting RAG platform at ridge.karthikjayan.tech), never generic dictionary or geological definitions. "
    "Stay strictly grounded in your verified experience, projects, and background from karthikjayan.dev."
)


def chat(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for item in history:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                u, a = item
                if u:
                    messages.append({"role": "user", "content": str(u)})
                if a:
                    messages.append({"role": "assistant", "content": str(a)})
    messages.append({"role": "user", "content": message})

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=300,
        temperature=0.2,
        stream=True
    )

    partial_text = ""
    for chunk in response:
        delta = chunk["choices"][0].get("delta", {}).get("content", "")
        partial_text += delta
        yield partial_text


demo = gr.ChatInterface(
    fn=chat,
    title="⚡ Karthik Jayan Portfolio AI",
    description="Dedicated SLM fine-tuned on verified engineering systems & background from [karthikjayan.dev](https://karthikjayan.dev).",
    examples=[
        "What is Ridge?",
        "Tell me about your software background.",
        "Explain your Recall project.",
        "What is your tech stack?"
    ],
    theme="ocean"
)

if __name__ == "__main__":
    demo.launch()
