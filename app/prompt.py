def build_prompt(
    user_question: str,
    mode: str,
    style: str,
    retrieved_context: str,
    history: list | None = None,
    crisis_banner: str | None = None
):
    """
    mode: "Policy/Admin" | "Study Skills" | "Identity & Accessibility"
    style: "Summary" | "Checklist" | "Detailed"
    """
    sys = (
        "You are an RMIT Study & Policy Advisor for the School of Computing Technologies. "
        "Use ONLY the provided source excerpts. Summarize accurately; do not invent policy. "
        "Always include brief source citations with title and page/section where applicable. "
        "If the evidence is insufficient, state what is missing and suggest the user upload the relevant document."
    )

    mode_inst_map = {
        "Policy/Admin": "Focus on RMIT policies, procedures, eligibility, definitions, and student steps.",
        "Study Skills": "Focus on practical study strategies, Learning Lab/library resources, time management, writing and groupwork tips.",
        "Identity & Accessibility": (
            "Prioritize identity, LGBTIQA+ and accessibility topics: ELS/ELP processes, gender affirmation steps, "
            "reasonable adjustments, safer community pathways, counselling and wellbeing services. "
            "Be respectful, non-diagnostic, and signpost official supports."
        )
    }
    style_inst_map = {
        "Summary": "Provide a concise bullet-point summary and a short citation list.",
        "Checklist": "Provide a step-by-step checklist with actions, required documents, timelines, and citations.",
        "Detailed": "Provide a detailed, structured explanation with subheadings and citations."
    }

    mode_inst = mode_inst_map.get(mode, mode_inst_map["Policy/Admin"])
    style_inst = style_inst_map.get(style, style_inst_map["Summary"])

    hist = ""
    if history:
        for h in history[-4:]:
            hist += f"\nUser: {h.get('user','')}\nAssistant: {h.get('bot','')}"

    banner = (crisis_banner + "\n\n") if crisis_banner else ""
    prompt = (
        f"System:\n{sys}\n\n"
        f"Mode instruction: {mode_inst}\n"
        f"Style instruction: {style_inst}\n\n"
        f"{banner}"
        f"Context (authoritative excerpts):\n{retrieved_context}\n\n"
        f"{hist}\n"
        f"User question:\n{user_question}\n"
        f"Answer:"
    )
    return prompt
