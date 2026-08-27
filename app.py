import streamlit as st
import time
import html

from rag_pipeline import answer_question


# ============================================================
# CONFIG
# ============================================================

BUSINESS_NAME = "SmileCare Dental Clinic"
ASSISTANT_NAME = "SmileCare AI Assistant"
ASSISTANT_TAGLINE = "Your friendly dental clinic assistant, online 24/7"
ICON = "🦷"

SUGGESTED_QUESTIONS = [
    (
        "🦷",
        "What services do you offer?",
        "What dental services does SmileCare provide?",
    ),
    (
        "📅",
        "How can I book an appointment?",
        "How can I book an appointment?",
    ),
    (
        "🕐",
        "What are your opening hours?",
        "What are the clinic opening hours?",
    ),
    (
        "💳",
        "What insurance do you accept?",
        "What insurance plans does SmileCare accept?",
    ),
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=ASSISTANT_NAME,
    page_icon=ICON,
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');


/* ============================================================
   GLOBAL
   ============================================================ */

html,
body,
[class*="css"] {
    font-family: "Inter", sans-serif;
}

/*
   Force a readable default text color everywhere.

   Streamlit Community Cloud can auto-switch to a dark theme
   based on the visitor's browser/OS setting. Without this,
   default text (anything not explicitly colored below)
   inherits white-on-white-ish behavior against our cream
   background whenever a visitor has dark mode on.
*/

.stApp,
.stApp p,
.stApp span,
.stApp div,
.stMarkdown {
    color: #263832;
}

.stApp {
    background:
        radial-gradient(
            circle at 10% 5%,
            rgba(27, 138, 107, 0.08),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(15, 92, 76, 0.06),
            transparent 30%
        ),
        linear-gradient(
            180deg,
            #FBF8F1 0%,
            #F3F0E7 100%
        );

    min-height: 100vh;
}


/* ============================================================
   HIDE STREAMLIT DEFAULT UI
   ============================================================ */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* ============================================================
   MAIN CONTAINER
   ============================================================ */

.block-container {
    max-width: 880px;
    padding-top: 2rem;
    padding-bottom: 7rem;
}


/* ============================================================
   HERO
   ============================================================ */

.hero {
    text-align: center;
    padding: 10px 0 6px;
    animation: heroIn 0.7s ease-out;
}

.logo {
    width: 78px;
    height: 78px;

    margin: 0 auto 16px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 24px;

    background:
        linear-gradient(
            135deg,
            #0F5C4C 0%,
            #1B8A6B 100%
        );

    font-size: 36px;

    box-shadow:
        0 16px 36px rgba(15, 92, 76, 0.22);

    animation:
        logoFloat 4s ease-in-out infinite;
}

.hero-title {
    font-family: "Fraunces", serif;

    font-size: 2.1rem;
    font-weight: 600;

    color: #1C2B26;

    letter-spacing: -0.01em;
}

.hero-subtitle {
    margin-top: 6px;

    color: #66756E;

    font-size: 1rem;
}

.status {
    display: inline-flex;

    align-items: center;

    gap: 7px;

    margin-top: 14px;

    padding: 6px 13px;

    border-radius: 999px;

    background:
        rgba(255, 255, 255, 0.7);

    border:
        1px solid rgba(27, 138, 107, 0.25);

    color: #0F5C4C;

    font-size: 0.76rem;

    font-weight: 600;
}

.status-dot {
    width: 7px;
    height: 7px;

    border-radius: 50%;

    background: #1B8A6B;

    animation:
        pulse 2s infinite;
}


/* ============================================================
   EMPTY STATE
   ============================================================ */

.empty-state {
    position: relative;

    margin-top: 40px;

    padding: 38px 32px;

    text-align: center;

    border-radius: 24px;

    background:
        rgba(255, 255, 255, 0.75);

    border:
        1px solid rgba(255, 255, 255, 0.9);

    box-shadow:
        0 20px 55px rgba(28, 43, 38, 0.06);

    backdrop-filter: blur(14px);

    animation:
        cardIn 0.8s ease-out;
}

.empty-icon {
    width: 58px;
    height: 58px;

    margin: 0 auto 14px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 18px;

    background:
        rgba(27, 138, 107, 0.10);

    font-size: 27px;

    animation:
        iconFloat 3.5s ease-in-out infinite;
}

.empty-title {
    font-family: "Fraunces", serif;

    color: #1C2B26;

    font-size: 1.25rem;

    font-weight: 600;
}

.empty-description {
    max-width: 520px;

    margin: 10px auto 0;

    color: #66756E;

    font-size: 0.92rem;

    line-height: 1.65;
}


/* ============================================================
   SECTION LABEL
   ============================================================ */

.section-label {
    margin: 28px 0 12px;

    text-align: center;

    color: #8A9891;

    font-size: 0.7rem;

    font-weight: 700;

    text-transform: uppercase;

    letter-spacing: 0.1em;
}


/* ============================================================
   SUGGESTED QUESTION BUTTONS
   ============================================================ */

.stButton > button {
    min-height: 50px;

    border-radius: 14px !important;

    border:
        1px solid rgba(27, 138, 107, 0.18) !important;

    background:
        rgba(255, 255, 255, 0.82) !important;

    color: #30443B !important;

    font-size: 0.85rem !important;

    font-weight: 500 !important;

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease,
        border-color 0.2s ease,
        background 0.2s ease !important;
}

.stButton > button:hover {
    transform:
        translateY(-2px);

    border-color:
        rgba(27, 138, 107, 0.4) !important;

    background:
        rgba(255, 255, 255, 0.98) !important;

    box-shadow:
        0 10px 24px rgba(15, 92, 76, 0.10);
}

.stButton > button:active {
    transform:
        translateY(0px) scale(0.98);
}


/* ============================================================
   CHAT MESSAGE CONTAINER
   ============================================================ */

[data-testid="stChatMessage"] {
    padding: 10px 12px;

    border-radius: 18px;

    animation:
        msgIn 0.3s ease-out;
}


/* ============================================================
   CHAT MESSAGE TEXT
   ============================================================ */

/*
   Dark green/charcoal instead of Streamlit's
   default text color.

   This is specifically designed to remain
   clearly visible against the cream background.
*/

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: #263832 !important;

    line-height: 1.65;

    font-size: 0.93rem;
}


/* ============================================================
   BOLD TEXT
   ============================================================ */

[data-testid="stChatMessage"] strong {
    color: #173D32 !important;

    font-weight: 600;
}


/* ============================================================
   MARKDOWN HEADINGS
   ============================================================ */

[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] h4 {
    color: #173D32 !important;
}


/* ============================================================
   LINKS
   ============================================================ */

[data-testid="stChatMessage"] a {
    color: #0F6B55 !important;
}


/* ============================================================
   CODE
   ============================================================ */

[data-testid="stChatMessage"] code {
    color: #263832 !important;

    background:
        #EDE9DE !important;

    border-radius: 5px;

    padding: 2px 5px;
}


/* ============================================================
   CHAT INPUT
   ============================================================ */

[data-testid="stChatInput"] {
    animation:
        fadeUp 0.6s ease-out;
}

[data-testid="stChatInput"] textarea {
    min-height: 52px !important;

    border-radius: 16px !important;

    border:
        1px solid #E4E0D4 !important;

    background:
        rgba(255, 255, 255, 0.95) !important;

    color: #263832 !important;

    caret-color: #1B8A6B;

    box-shadow:
        0 8px 28px rgba(28, 43, 38, 0.07);

    transition:
        border-color 0.2s ease,
        box-shadow 0.2s ease !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #9AA69F !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color:
        #1B8A6B !important;

    box-shadow:
        0 0 0 3px rgba(27, 138, 107, 0.12),
        0 10px 30px rgba(28, 43, 38, 0.09);
}


/* ============================================================
   SOURCES
   ============================================================ */

.sources-container {
    margin-top: 10px;

    margin-bottom: 18px;

    animation:
        srcIn 0.4s ease-out;
}

.sources-label {
    margin-bottom: 7px;

    color: #8A9891;

    font-size: 0.66rem;

    font-weight: 700;

    text-transform: uppercase;

    letter-spacing: 0.09em;
}

.source-card {
    display: inline-flex;

    align-items: center;

    gap: 6px;

    margin-right: 6px;

    margin-bottom: 5px;

    padding: 6px 10px;

    border-radius: 9px;

    background:
        rgba(255, 255, 255, 0.82);

    border:
        1px solid #E4E0D4;

    color: #53645C;

    font-size: 0.72rem;

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}

.source-card:hover {
    transform:
        translateY(-2px);

    box-shadow:
        0 5px 14px rgba(28, 43, 38, 0.06);
}


/* ============================================================
   TYPING INDICATOR
   ============================================================ */

.typing-dots {
    display: flex;

    align-items: center;

    gap: 3px;

    height: 20px;
}

.typing-dots span {
    display: inline-block;

    width: 6px;
    height: 6px;

    border-radius: 50%;

    background:
        #1B8A6B;

    animation:
        typingBounce 1.2s infinite ease-in-out;
}

.typing-dots span:nth-child(2) {
    animation-delay:
        0.15s;
}

.typing-dots span:nth-child(3) {
    animation-delay:
        0.3s;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;

    margin-top: 55px;

    color: #9AA69F;

    font-size: 0.72rem;

    line-height: 1.8;

    animation:
        fadeUp 1s ease-out;
}

.footer strong {
    color: #66756E;
}


/* ============================================================
   ANIMATIONS
   ============================================================ */

@keyframes heroIn {

    from {
        opacity: 0;
        transform: translateY(-18px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}


@keyframes cardIn {

    from {
        opacity: 0;
        transform:
            translateY(22px)
            scale(0.98);
    }

    to {
        opacity: 1;
        transform:
            translateY(0)
            scale(1);
    }
}


@keyframes msgIn {

    from {
        opacity: 0;
        transform:
            translateY(8px);
    }

    to {
        opacity: 1;
        transform:
            translateY(0);
    }
}


@keyframes srcIn {

    from {
        opacity: 0;
        transform:
            translateY(4px);
    }

    to {
        opacity: 1;
        transform:
            translateY(0);
    }
}


@keyframes fadeUp {

    from {
        opacity: 0;
        transform:
            translateY(12px);
    }

    to {
        opacity: 1;
        transform:
            translateY(0);
    }
}


@keyframes logoFloat {

    0%,
    100% {
        transform:
            translateY(0);
    }

    50% {
        transform:
            translateY(-6px);
    }
}


@keyframes iconFloat {

    0%,
    100% {
        transform:
            translateY(0);
    }

    50% {
        transform:
            translateY(-4px);
    }
}


@keyframes pulse {

    0% {
        box-shadow:
            0 0 0 0
            rgba(27, 138, 107, 0.45);
    }

    70% {
        box-shadow:
            0 0 0 6px
            rgba(27, 138, 107, 0);
    }

    100% {
        box-shadow:
            0 0 0 0
            rgba(27, 138, 107, 0);
    }
}


@keyframes typingBounce {

    0%,
    60%,
    100% {
        transform:
            translateY(0);

        opacity:
            0.5;
    }

    30% {
        transform:
            translateY(-4px);

        opacity:
            1;
    }
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 640px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero-title {
        font-size: 1.7rem;
    }

    .hero-subtitle {
        font-size: 0.9rem;
    }

    .empty-state {
        margin-top: 28px;

        padding:
            28px 18px;
    }

    .empty-description {
        font-size: 0.87rem;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPER — RENDER SOURCES
# ============================================================

def render_sources(sources):

    if not sources:
        return

    cards = ""

    for source in sources:

        name = html.escape(
            source.get(
                "source",
                "Unknown source",
            )
        )

        doc_type = html.escape(
            source.get(
                "document_type",
                "general",
            )
        )

        cards += (
            '<span class="source-card">'
            '📄 '
            f'<span>{name} · {doc_type}</span>'
            '</span>'
        )

    st.markdown(
        '<div class="sources-container">'
        '<div class="sources-label">'
        'Knowledge sources'
        '</div>'
        f'{cards}'
        '</div>',
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
    f'<div class="hero-title">'
    f'{html.escape(ASSISTANT_NAME)}'
    f'</div>'
    f'<div class="hero-subtitle">'
    f'{html.escape(ASSISTANT_TAGLINE)}'
    f'</div>'
    f'<div class="status">'
    f'<span class="status-dot"></span>'
    f'AI Assistant Online'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)


# ============================================================
# NEW CHAT BUTTON
# ============================================================

if st.session_state.messages:

    col1, col2, col3 = st.columns(
        [1, 2, 1]
    )

    with col2:

        if st.button(
            "✦  Start a new conversation",
            use_container_width=True,
        ):

            st.session_state.messages = []

            st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message["role"]

    avatar = (
        ICON
        if role == "assistant"
        else "👤"
    )

    with st.chat_message(
        role,
        avatar=avatar,
    ):

        st.markdown(
            message["content"]
        )

        if role == "assistant":

            render_sources(
                message.get("sources")
            )


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.messages:

    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-icon">💬</div>'
        '<div class="empty-title">'
        'How can I help you today?'
        '</div>'
        f'<div class="empty-description">'
        f'Ask about '
        f'{html.escape(BUSINESS_NAME)}'
        f'\'s services, appointments, '
        f'opening hours, insurance, or policies '
        f'— I\'ll answer directly from the clinic\'s '
        f'knowledge base.'
        f'</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-label">'
        'Popular questions'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(
        2,
        gap="small",
    )

    columns = [
        col1,
        col2,
        col1,
        col2,
    ]

    for (
        emoji,
        label,
        full_question,
    ), col in zip(
        SUGGESTED_QUESTIONS,
        columns,
    ):

        with col:

            if st.button(
                f"{emoji}  {label}",
                use_container_width=True,
            ):

                st.session_state.pending_question = (
                    full_question
                )

                st.rerun()


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    f"Ask {BUSINESS_NAME} a question..."
)


# ============================================================
# HANDLE SUGGESTED QUESTIONS
# ============================================================

if "pending_question" in st.session_state:

    question = (
        st.session_state.pending_question
    )

    del st.session_state.pending_question


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    question = question.strip()

    if question:

        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message(
            "user",
            avatar="👤",
        ):

            st.markdown(question)


        # ----------------------------------------------------
        # ASSISTANT MESSAGE
        # ----------------------------------------------------

        with st.chat_message(
            "assistant",
            avatar=ICON,
        ):

            # ------------------------------------------------
            # TYPING INDICATOR
            # ------------------------------------------------

            typing_placeholder = st.empty()

            typing_placeholder.markdown(
                '<div class="typing-dots">'
                '<span></span>'
                '<span></span>'
                '<span></span>'
                '</div>',
                unsafe_allow_html=True,
            )


            # ------------------------------------------------
            # RUN RAG PIPELINE
            # ------------------------------------------------

            try:

                history = (
                    st.session_state.messages[:-1]
                )

                result = answer_question(
                    question=question,
                    conversation_history=history,
                )

                answer = result.get(
                    "answer",
                    "I'm sorry, I couldn't generate an answer.",
                )

                sources = result.get(
                    "sources",
                    [],
                )

            except Exception as error:

                answer = (
                    "I'm sorry, something went wrong "
                    "while processing your question. "
                    "Please try again in a moment."
                )

                sources = []

                print(
                    f"RAG pipeline error: {error}"
                )


            # ------------------------------------------------
            # REMOVE TYPING INDICATOR
            # ------------------------------------------------

            typing_placeholder.empty()


            # ------------------------------------------------
            # SMOOTH RESPONSE REVEAL
            # ------------------------------------------------

            reveal_placeholder = st.empty()

            words = answer.split()

            displayed = ""

            for index, word in enumerate(words):

                displayed += word

                if index < len(words) - 1:

                    displayed += " "

                reveal_placeholder.markdown(
                    displayed
                )

                time.sleep(0.012)


            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            render_sources(
                sources
            )


        # ----------------------------------------------------
        # SAVE ASSISTANT RESPONSE
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f'<div class="footer">'
    f'{ICON} '
    f'<strong>'
    f'{html.escape(BUSINESS_NAME)}'
    f'</strong>'
    f'<br>'
    f'AI-powered customer support assistant'
    f'<br>'
    f'Answers grounded in the clinic\'s knowledge base'
    f'<br><br>'
    f'Powered by <strong>AI Agents</strong>'
    f'</div>',
    unsafe_allow_html=True,
)