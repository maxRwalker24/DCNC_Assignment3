import os
import pickle
import gradio as gr

from app.retrieve import retrieve, format_context
from app.prompt import build_prompt
from app.bedrock_invoke import invoke_bedrock
from app.session_ingest import SessionMiniIndex, extract_text_from_uploads

# Load base vectorizer for session mini-index
with open("data/vectorizer.pkl", "rb") as f:
    BASE_V = pickle.load(f)

DISTRESS_KEYWORDS = [
    "panic", "crisis", "unsafe", "suicid", "self-harm", "self harm",
    "harm myself", "ending my life", "hurt myself", "sexual harm",
    "assault", "harassment", "stalking"
]

def detect_crisis_banner(user_text: str):
    t = (user_text or "").lower()
    if any(k in t for k in DISTRESS_KEYWORDS):
        # Minimal banner; you can customize with local numbers if you have exact citations in KB
        return (
            "If you need urgent help now, contact **RMIT Urgent Mental Health Support** "
            "(after hours phone 1300 305 737 or text 0488 884 162). "
            "For safety concerns you can also contact **Safer Community** or **RMIT Counselling** during business hours."
        )
    return None

def mode_to_tags(mode: str):
    if mode == "Policy/Admin":
        return ["assessment", "extensions", "special", "appeals", "census", "enrolment", "integrity", "misconduct", "credit"]
    if mode == "Study Skills":
        return ["study", "time", "writing", "groupwork"]
    if mode == "Identity & Accessibility":
        return ["els", "elp", "accessibility", "neurodivergent", "gender-affirmation", "lgbtiqa", "wellbeing", "safer-community", "wil"]
    return []

def app_fn(question, mode, style, focus_tags, files, include_uploads, history):
    # 1) Guardrail banner
    crisis_banner = detect_crisis_banner(question)

    # 2) Retrieval over base KB with tag boosts (from mode + user focus)
    boosts = list(dict.fromkeys(mode_to_tags(mode) + (focus_tags or [])))
    kb_hits = retrieve(question, top_k=6, tag_boost=boosts)

    # 3) Optional session uploads
    upload_ctx = ""
    if include_uploads and files:
        mini = SessionMiniIndex(BASE_V)
        texts = extract_text_from_uploads(files)
        mini.add_upload_texts(texts)
        up_hits = mini.retrieve(question, top_k=3)
        if up_hits:
            blocks = []
            for score, cid, text in up_hits:
                blocks.append(f"### Source: User Upload\n{text}\n")
            upload_ctx = "\n\n".join(blocks)

    # 4) Build context
    kb_ctx = format_context(kb_hits)
    context = (kb_ctx + ("\n\n" + upload_ctx if upload_ctx else "")).strip()

    # 5) Prompt & invoke
    prompt = build_prompt(
        user_question=question,
        mode=mode,
        style=style,
        retrieved_context=context,
        history=history or [],
        crisis_banner=crisis_banner
    )
    answer = invoke_bedrock(prompt)

    # 6) Update history
    history = (history or []) + [{"user": question, "bot": answer}]

    # 7) Sources list
    sources = []
    for score, cid, m in kb_hits:
        title = m.get("source_title") or "Source"
        a, b = m.get("page_start"), m.get("page_end")
        if a and b and a == b:
            pages = f" (p. {a})"
        elif a and b:
            pages = f" (pp. {a}–{b})"
        else:
            pages = ""
        sources.append(f"- {title}{pages}")
    if upload_ctx:
        sources.append("- User Upload")

    footer = "Sources used:\n" + "\n".join(dict.fromkeys(sources))
    return f"{answer}\n\n---\n{footer}", history

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 🎓 RMIT Study & Policy Advisor")

        with gr.Row():
            mode = gr.Radio(
                ["Policy/Admin", "Study Skills", "Identity & Accessibility"],
                value="Policy/Admin",
                label="Mode"
            )
            style = gr.Radio(
                ["Summary", "Checklist", "Detailed"],
                value="Summary",
                label="Response Style"
            )

        focus = gr.CheckboxGroup(
            ["integrity","extensions","special","appeals","census","enrolment",
             "assessment","writing","study","time","groupwork",
             "els","elp","accessibility","neurodivergent","gender-affirmation","lgbtiqa","wellbeing","safer-community","wil"],
            label="Focus (boost retrieval)",
            value=["assessment"]
        )

        question = gr.Textbox(
            label="Your question",
            placeholder="e.g., Am I eligible for an assessment extension? What proof do I need?",
            lines=2
        )

        uploads = gr.File(
            label="Upload documents (optional)",
            file_types=[".pdf", ".txt"],
            file_count="multiple"
        )
        include_uploads = gr.Checkbox(
            label="Include uploaded documents in answers",
            value=True
        )

        out = gr.Textbox(label="Answer", lines=18)
        state = gr.State([])

        submit = gr.Button("Get Advice")
        submit.click(
            app_fn,
            inputs=[question, mode, style, focus, uploads, include_uploads, state],
            outputs=[out, state]
        )

    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
