# formatter.py

import streamlit as st

def render_claim_analysis(parsed, result):
    st.subheader("Denial Risk Score")

    score = result.get("risk_score", 0)
    if score >= 70:
        label = "High Risk"
    elif score >= 40:
        label = "Moderate Risk"
    else:
        label = "Low Risk"

    st.markdown(f"**Score:** {score}% — {label}")

def render_agent_trace(parsed, result):
    import streamlit as st
    from tools import cms_tools

    st.subheader("Agent Decision Trace")

    st.markdown("**Structured Input Passed to Tool:**")
    st.code(parsed.get("action_input", ""), language="text")

    st.markdown("**Available Tools:**")
    selected_tool = parsed.get("action", "")
    for tool in cms_tools:
        label = f"**[Selected] {tool.name}**" if tool.name == selected_tool else tool.name
        st.markdown(f"- {label}")

    st.markdown("**Agent Reasoning (LLM Thought):**")
    st.markdown(parsed.get("thought", "_No explanation provided._"))

    st.markdown("**Raw Tool Result:**")
    st.json(result)



def render_retrieved_policy_docs(tool_name, result):
    if tool_name != "RetrieveCMSPolicy":
        return

    documents = result if isinstance(result, list) else []
    if not documents:
        return

    with st.expander("Retrieved CMS Billing Policy", expanded=False):
        for doc in documents:
            content = getattr(doc, "page_content", None) or str(doc)
            st.markdown(content)
