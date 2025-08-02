import re
from langchain_ollama import OllamaLLM
from tools import cms_tools  # <- contains AnalyzeClaim, RetrieveCMSPolicy

# 🧠 Set up Mistral (via Ollama)
llm = OllamaLLM(model="mistral:instruct", temperature=0.4)  # 🔒 Lower temp for more deterministic output

# 📜 Prompt instructions: Now explicitly block JSON
prompt_text = """
You are a denial risk analysis assistant. Your ONLY valid response format is XML with exactly these three tags:

<output>
  <thought>[Brief explanation of your reasoning]</thought>
  <action>[Choose either AnalyzeClaim or RetrieveCMSPolicy]</action>
  <action_input>[Plain string only — do NOT use quotes, braces, or extra commentary]</action_input>
</output>

⚠️ DO NOT return JSON, markdown, bullet points, or explanations outside the <output> block.
Return only one complete <output> block and nothing else.
"""

# 🧽 Clean LLM quirks
def sanitize_llm_output(output: str) -> str:
    # Trim after </output> to remove any trailing commentary or citations
    if "</output>" in output:
        output = output.split("</output>")[0] + "</output>"
    return output.replace("{", "").replace("}", "").replace('"', "").strip()

# ❌ Check for accidental JSON
def looks_like_json(text: str) -> bool:
    return text.strip().startswith("{") or "```json" in text or "risk_score" in text.lower()

# 🔍 Soft XML parser
def loose_parse(llm_output: str):
    try:
        thought = re.search(r"<thought>(.*?)</thought>", llm_output, re.DOTALL).group(1).strip()
        action = re.search(r"<action>(.*?)</action>", llm_output, re.DOTALL).group(1).strip()
        action_input = re.search(r"<action_input>(.*?)</action_input>", llm_output, re.DOTALL).group(1).strip()
        return {"thought": thought, "action": action, "action_input": action_input}
    except:
        return None

# 🧠 Agent logic
def generate_agent_decision(user_query):
    full_prompt = f"{prompt_text}\n\n{user_query}"
    raw_output = llm.invoke(full_prompt)

    if looks_like_json(raw_output):
        print("\n❌ Unexpected JSON format detected. Skipping parse.\n")
        print(raw_output)
        return None

    cleaned = sanitize_llm_output(raw_output)
    parsed = loose_parse(cleaned)

    if not parsed:
        print("\n❌ Parsing failed. Raw LLM output:\n", raw_output)
    return parsed

# 🔁 Run full agent
def run_agent(user_query):
    parsed = generate_agent_decision(user_query)

    if not parsed:
        print("⛔ No actionable result. Stopping.")
        return

    print("\n🤖 Agent Output:")
    print("💡 Thought:", parsed["thought"])
    print("🛠️ Action:", parsed["action"])
    print("📥 Input :", parsed["action_input"])

    tool_map = {tool.name: tool for tool in cms_tools}
    tool_name = parsed["action"]
    if tool_name not in tool_map:
        raise ValueError(f"❌ Unknown tool: '{tool_name}'")

    tool = tool_map[tool_name]
    result = tool.func(parsed["action_input"])
    print("\n✅ Tool Result:\n", result)

# 🧪 Example tests
if __name__ == "__main__":
    print("\n🩺 Test 1: Denial check")
    run_agent("Would CPT 99214 with diagnosis Z79.899 and modifier -25 be denied under Medicare?")

    print("\n💬 Test 2: Diagnosis validity")
    run_agent("Is Z79.899 a valid diagnosis for Medicare claims?")

    print("\n📜 Test 3: CMS rule for modifier -25")
    run_agent("What CMS guidance exists about denials involving modifier -25?")
