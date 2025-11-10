# app/chatbot_ui.py
"""
Gradio UI for the RMIT Student Support Chatbot.

Flow:
- User enters a question and chooses an assistant role (default = Auto).
- We build retrieval context + LLM prompt via `get_context_and_prompt(...)`.
- We call Bedrock (Claude) with the prompt and render the reply.
- We also show the retrieved context and citations for transparency.

This file keeps the UI minimal and assignment-friendly.
"""

import gradio as gr

from app.session_ingest import get_context_and_prompt, CATEGORY_MAP
from app.bedrock_invoke import invoke_bedrock


# --- UI choices --------------------------------------------------------------

# Add an "Auto" option that detects the best category from the query.
AUTO_LABEL = "Auto (recommended)"
DEFAULT_CATEGORY = AUTO_LABEL

# Existing explicit roles (as seen by the user).
CATEGORIES = list(CATEGORY_MAP.keys())

# Final dropdown options shown in the UI (Auto first, then explicit roles).
CHOICES = [AUTO_LABEL] + CATEGORIES


# --- Core callback -----------------------------------------------------------

def answer_fn(user_query: str, category_label: str, k: int, temperature: float, top_p: float):
    """
    Main UI callback. Given user inputs, produce:
      - the assistant's answer,
      - the context snippets used,
      - the citations list.

    Returns a 3-tuple to populate the three output textboxes.
    """
    if not user_query or not user_query.strip():
        # Keep the UI friendly if the user submits an empty question.
        return "Please enter a question.", "", ""

    # 1) Build retrieval context + final LLM prompt
    ctx, cites, prompt = get_context_and_prompt(
        user_query=user_query.strip(),
        ui_category_label=category_label,
        top_k=int(k),
        max_ctx_chars=2800,
    )

    # If retrieval found nothing, we still proceed but warn in the context pane.
    if not ctx.strip():
        ctx = "(No matching context found — answer cautiously or ask the user to rephrase.)"

    # 2) Call Bedrock (Claude 3 via the simple wrapper)
    model_reply = invoke_bedrock(
        prompt_text=prompt,
        max_tokens=700,
        temperature=float(temperature),
        top_p=float(top_p),
    )

    # 3) Prepare citations block (one-per-source, already formatted upstream)
    citations_text = "\n".join(cites) if cites else "No citations (context empty)."

    # Gradio wires these to the three output components (Answer, Context, Citations)
    return model_reply, ctx, citations_text


# --- UI assembly -------------------------------------------------------------

def build_ui():
    """
    Construct and return the Gradio Blocks UI.
    """
    with gr.Blocks(title="RMIT Student Support Chatbot") as demo:
        # Title + short description
        gr.Markdown("# 🎓 RMIT Student Support Chatbot")
        gr.Markdown(
            "Ask questions about **policies/admin**, **study skills**, or **identity & accessibility**.\n\n"
            "The bot retrieves relevant snippets from your local corpus and cites sources as `[S#]`."
        )

        # Controls row: role (with Auto), retrieval top-K, sampling params
        with gr.Row():
            category = gr.Dropdown(
                choices=CHOICES,
                value=DEFAULT_CATEGORY,
                label="Assistant Role",
            )
            k = gr.Slider(1, 8, value=4, step=1, label="Snippets (Top-K)")
            temperature = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")

        # Main question input
        user_query = gr.Textbox(
            lines=3,
            label="Your question",
            placeholder="e.g., How do I request an assessment extension for illness?",
        )

        # Submit button
        submit = gr.Button("💡 Get Advice")

        # Output tabs: Answer | Context | Citations
        with gr.Tab("Answer"):
            answer = gr.Textbox(lines=12, label="Assistant Answer")

        with gr.Tab("Retrieved context (for transparency)"):
            context = gr.Textbox(lines=18, label="Context snippets")

        with gr.Tab("Citations"):
            citations = gr.Textbox(lines=10, label="Sources used")

        # Wire the button to the callback
        submit.click(
            fn=answer_fn,
            inputs=[user_query, category, k, temperature, top_p],
            outputs=[answer, context, citations],
        )

    return demo


# --- Entrypoint --------------------------------------------------------------

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860)


