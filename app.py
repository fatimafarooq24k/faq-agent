import streamlit as st
import time
import html

from rag_pipeline import answer_question


# ============================================================
# CONFIG — change these to re-skin for a new client in minutes
# ============================================================

BUSINESS_NAME = "SmileCare Dental Clinic"
ASSISTANT_NAME = "SmileCare AI Assistant"
ASSISTANT_TAGLINE = "Your friendly dental clinic assistant, online 24/7"
ICON = "🦷"
SUGGESTED_QUESTIONS = [
    ("🦷", "What services do you offer?", "What dental services does SmileCare provide?"),
    ("📅", "How can I book an appointment?", "How can I book an appointment?"),
    ("🕐", "What are your opening hours?", "What are the clinic opening hours?"),
    ("💳", "What insurance do you accept?", "What insurance plans does SmileCare accept?"),
]


st.set_page_config(
    page_title=ASSISTANT_NAME,
    page_icon=ICON,
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# STYLES — flattened strings, no leading whitespace anywhere
# (indented HTML inside triple-quoted strings gets rendered as
# a literal Markdown code block, which is a common Streamlit bug)
# ============================================================

st.markdown(
'<style>'
'@import url(\'https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Inter:wght@400;500;600&display=swap\');'
'html, body, [class*="css"] { font-family: "Inter", sans-serif; }'
'.stApp { background: linear-gradient(180deg, #FBF8F1 0%, #F3F0E7 100%); min-height: 100vh; }'
'#MainMenu { visibility: hidden; } footer { visibility: hidden; } header { background: transparent !important; }'
'.block-container { max-width: 880px; padding-top: 2rem; padding-bottom: 7rem; }'

'.hero { text-align: center; padding: 10px 0 6px; animation: heroIn 0.7s ease-out; }'
'.logo { width: 78px; height: 78px; margin: 0 auto 16px; display: flex; align-items: center; justify-content: center; '
'border-radius: 24px; background: linear-gradient(135deg, #0F5C4C 0%, #1B8A6B 100%); font-size: 36px; '
'box-shadow: 0 16px 36px rgba(15,92,76,0.22); animation: logoFloat 4s ease-in-out infinite; }'
'.hero-title { font-family: "Fraunces", serif; font-size: 2.1rem; font-weight: 600; color: #1C2B26; letter-spacing: -0.01em; }'
'.hero-subtitle { margin-top: 6px; color: #6B7A73; font-size: 1rem; }'
'.status { display: inline-flex; align-items: center; gap: 7px; margin-top: 14px; padding: 6px 13px; border-radius: 999px; '
'background: rgba(255,255,255,0.7); border: 1px solid rgba(27,138,107,0.25); color: #0F5C4C; font-size: 0.76rem; font-weight: 600; }'
'.status-dot { width: 7px; height: 7px; border-radius: 50%; background: #1B8A6B; animation: pulse 2s infinite; }'

'.empty-state { position: relative; margin-top: 40px; padding: 38px 32px; text-align: center; border-radius: 24px; '
'background: rgba(255,255,255,0.75); border: 1px solid rgba(255,255,255,0.9); box-shadow: 0 20px 55px rgba(28,43,38,0.06); '
'backdrop-filter: blur(14px); animation: cardIn 0.8s ease-out; }'
'.empty-icon { width: 58px; height: 58px; margin: 0 auto 14px; display: flex; align-items: center; justify-content: center; '
'border-radius: 18px; background: rgba(27,138,107,0.10); font-size: 27px; animation: iconFloat 3.5s ease-in-out infinite; }'
'.empty-title { font-family: "Fraunces", serif; color: #1C2B26; font-size: 1.25rem; font-weight: 600; }'
'.empty-description { max-width: 520px; margin: 10px auto 0; color: #6B7A73; font-size: 0.92rem; line-height: 1.65; }'
'.section-label { margin: 28px 0 12px; text-align: center; color: #9AA69F; font-size: 0.7rem; font-weight: 700; '
'text-transform: uppercase; letter-spacing: 0.1em; }'

'.stButton > button { min-height: 50px; border-radius: 14px !important; border: 1px solid rgba(27,138,107,0.18) !important; '
'background: rgba(255,255,255,0.8) !important; color: #2A3B34 !important; font-size: 0.85rem !important; font-weight: 500 !important; '
'transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important; }'
'.stButton > button:hover { transform: translateY(-2px); border-color: rgba(27,138,107,0.4) !important; '
'box-shadow: 0 10px 24px rgba(15,92,76,0.10); }'

'[data-testid="stChatMessage"] { padding: 10px 12px; border-radius: 18px; animation: msgIn 0.3s ease-out; }'
'[data-testid="stChatMessage"] p { line-height: 1.6; font-size: 0.93rem; }'
'[data-testid="stChatInput"] textarea { min-height: 52px !important; border-radius: 16px !important; '
'border: 1px solid #E4E0D4 !important; background: rgba(255,255,255,0.95) !important; '
'box-shadow: 0 8px 28px rgba(28,43,38,0.07); transition: border-color 0.2s ease, box-shadow 0.2s ease !important; }'
'[data-testid="stChatInput"] textarea:focus { border-color: #1B8A6B !important; '
'box-shadow: 0 0 0 3px rgba(27,138,107,0.12), 0 10px 30px rgba(28,43,38,0.09); }'

'.sources-container { margin-top: 10px; margin-bottom: 18px; animation: srcIn 0.4s ease-out; }'
'.sources-label { margin-bottom: 7px; color: #9AA69F; font-size: 0.66rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.09em; }'
'.source-card { display: inline-flex; align-items: center; gap: 6px; margin-right: 6px; margin-bottom: 5px; padding: 6px 10px; '
'border-radius: 9px; background: rgba(255,255,255,0.8); border: 1px solid #E4E0D4; color: #6B7A73; font-size: 0.72rem; }'

'.typing-dots span { display: inline-block; width: 6px; height: 6px; margin-right: 3px; border-radius: 50%; '
'background: #1B8A6B; animation: typingBounce 1.2s infinite ease-in-out; }'
'.typing-dots span:nth-child(2) { animation-delay: 0.15s; } .typing-dots span:nth-child(3) { animation-delay: 0.3s; }'

'.footer { text-align: center; margin-top: 55px; color: #9AA69F; font-size: 0.72rem; line-height: 1.8; animation: fadeUp 1s ease-out; }'
'.footer strong { color: #6B7A73; }'

'@keyframes heroIn { from { opacity: 0; transform: translateY(-18px); } to { opacity: 1; transform: translateY(0); } }'
'@keyframes cardIn { from { opacity: 0; transform: translateY(22px) scale(0.98); } to { opacity: 1; transform: translateY(0) scale(1); } }'
'@keyframes msgIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }'
'@keyframes srcIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }'
'@keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }'
'@keyframes logoFloat { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }'
'@keyframes iconFloat { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-4px); } }'
'@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(27,138,107,0.45); } 70% { box-shadow: 0 0 0 6px rgba(27,138,107,0); } '
'100% { box-shadow: 0 0 0 0 rgba(27,138,107,0); } }'
'@keyframes typingBounce { 0%, 60%, 100% { transform: translateY(0); opacity: 0.5; } 30% { transform: translateY(-4px); opacity: 1; } }'

'@media (max-width: 640px) { .block-container { padding-left: 1rem; padding-right: 1rem; } '
'.hero-title { font-size: 1.7rem; } .empty-state { margin-top: 28px; padding: 28px 18px; } }'
'</style>',
unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def render_sources(sources):
    if not sources:
        return
    cards = ""
    for source in sources:
        name = html.escape(source.get("source", "Unknown source"))
        doc_type = html.escape(source.get("document_type", "general"))
        cards += (
            f'<span class="source-card">📄 <span>{name} · {doc_type}</span></span>'
        )
    st.markdown(
        f'<div class="sources-container">'
        f'<div class="sources-label">Knowledge sources</div>{cards}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# HERO
# ============================================================

st.markdown(
f'<div class="hero">'
f'<div class="logo">{ICON}</div>'
f'<div class="hero-title">{html.escape(ASSISTANT_NAME)}</div>'
f'<div class="hero-subtitle">{html.escape(ASSISTANT_TAGLINE)}</div>'
f'<div class="status"><span class="status-dot"></span>AI Assistant Online</div>'
f'</div>',
unsafe_allow_html=True,
)


# ============================================================
# NEW CHAT BUTTON
# ============================================================

if st.session_state.messages:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✦  Start a new conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    role = message["role"]
    avatar = ICON if role == "assistant" else "👤"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])
        if role == "assistant":
            render_sources(message.get("sources"))


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.messages:
    st.markdown(
    '<div class="empty-state">'
    '<div class="empty-icon">💬</div>'
    '<div class="empty-title">How can I help you today?</div>'
    f'<div class="empty-description">Ask about {html.escape(BUSINESS_NAME)}\'s services, appointments, '
    'opening hours, insurance, or policies — I\'ll answer directly from the clinic\'s knowledge base.</div>'
    '</div>',
    unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Popular questions</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="small")
    columns = [col1, col2, col1, col2]
    for (emoji, label, full_question), col in zip(SUGGESTED_QUESTIONS, columns):
        with col:
            if st.button(f"{emoji}  {label}", use_container_width=True):
                st.session_state.pending_question = full_question
                st.rerun()


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(f"Ask {BUSINESS_NAME} a question...")

if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:
    question = question.strip()

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)

        with st.chat_message("assistant", avatar=ICON):
            # Typing indicator while the pipeline runs
            typing_placeholder = st.empty()
            typing_placeholder.markdown(
                '<div class="typing-dots"><span></span><span></span><span></span></div>',
                unsafe_allow_html=True,
            )

            try:
                history = st.session_state.messages[:-1]
                result = answer_question(question=question, conversation_history=history)
                answer = result.get("answer", "I'm sorry, I couldn't generate an answer.")
                sources = result.get("sources", [])
            except Exception as error:
                answer = (
                    "I'm sorry, something went wrong while processing your question. "
                    "Please try again in a moment."
                )
                sources = []
                print(f"RAG pipeline error: {error}")

            typing_placeholder.empty()

            # Smooth word-by-word reveal instead of an instant dump of text
            reveal_placeholder = st.empty()
            words = answer.split()
            displayed = ""
            for index, word in enumerate(words):
                displayed += word + (" " if index < len(words) - 1 else "")
                reveal_placeholder.markdown(displayed)
                time.sleep(0.012)

            render_sources(sources)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
f'<div class="footer">'
f'{ICON} <strong>{html.escape(BUSINESS_NAME)}</strong><br>'
'AI-powered customer support assistant<br>'
'Answers grounded in the clinic\'s knowledge base<br><br>'
'Powered by <strong>AI Agents</strong>'
'</div>',
unsafe_allow_html=True,
)