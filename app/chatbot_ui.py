# app/chatbot_ui.py
import os
from pathlib import Path
import gradio as gr

from app.session_ingest import get_context_and_prompt, CATEGORY_MAP
from app.bedrock_invoke import invoke_bedrock

DEFAULT_CATEGORY = "Policy & Admin Guide"  # UI label
CATEGORIES = list(CATEGORY_MAP.keys())

def answer_fn(user_query, category_label, k, temperature, top_p):
    if not user_query or not user_query.strip():
        return gr.update(value="Please enter a question."), "", ""

    # Prepare context and prompt
    ctx, cites, prompt = get_context_and_prompt(
        user_query=user_query.strip(),
        ui_category_label=category_label,
        top_k=int(k),
        max_ctx_chars=2800
    )

    if not ctx.strip():
        ctx = "(No matching context found — answer cautiously or ask the user to rephrase.)"

    # Call Bedrock (Claude 3)
    model_reply = invoke_bedrock(
        prompt_text=prompt,
        max_tokens=700,
        temperature=float(temperature),
        top_p=float(top_p),
    )

    citations_text = "\n".join(cites) if cites else "No citations (context empty)."
    return model_reply, ctx, citations_text


def build_ui():
    with gr.Blocks(title="RMIT Student Support Chatbot") as demo:
        gr.Markdown("# 🎓 RMIT Student Support Chatbot")
        gr.Markdown(
            "Ask questions about policies/admin, study skills, or identity & accessibility support.\n"
            "The bot retrieves matching snippets from your local corpus and cites sources as [S#]."
        )

        with gr.Row():
            category = gr.Dropdown(choices=CATEGORIES, value=DEFAULT_CATEGORY, label="Assistant Role")
            k = gr.Slider(1, 8, value=4, step=1, label="Snippets (Top-K)")
            temperature = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")

        user_query = gr.Textbox(
            lines=3,
            label="Your question",
            placeholder="e.g., How do I request an assessment extension for illness?"
        )

        submit = gr.Button("💡 Get Advice")

        with gr.Tab("Answer"):
            answer = gr.Textbox(lines=12, label="Assistant Answer")

        with gr.Tab("Retrieved context (for transparency)"):
            context = gr.Textbox(lines=18, label="Context snippets")

        with gr.Tab("Citations"):
            citations = gr.Textbox(lines=10, label="Sources used")

        submit.click(
            fn=answer_fn,
            inputs=[user_query, category, k, temperature, top_p],
            outputs=[answer, context, citations]
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
