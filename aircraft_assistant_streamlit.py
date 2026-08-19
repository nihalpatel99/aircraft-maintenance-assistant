import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from openai import AsyncOpenAI
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

load_dotenv()

st.set_page_config(page_title="Afircraft Maintenance Chatbot", page_icon="🤖", layout="centered")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
MODEL_DEPLOYMENT = os.getenv("MODEL_DEPLOYMENT")


@st.cache_resource
def get_client():
    """Create (and cache) the async Azure OpenAI client + credential for this session."""
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, "https://ai.azure.com/.default")
    client = AsyncOpenAI(base_url=AZURE_OPENAI_ENDPOINT, api_key=token_provider)
    return client, credential


def run_async(coro):
    """Run an async coroutine from Streamlit's synchronous script execution."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


async def get_response(client, prompt, previous_response_id, instructions):
    response = await client.responses.create(
        model=MODEL_DEPLOYMENT,
        instructions=instructions,
        input=prompt,
        previous_response_id=previous_response_id,
    )
    return response


# ---------------- Session state ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": "user"/"assistant", "content": str}, ...]
if "last_response_id" not in st.session_state:
    st.session_state.last_response_id = None

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    instructions = st.text_area(
        "System instructions",
        value="You are a helpful Aircraft Maintenance AI assistant that answers questions and provides information on diagnosis, risk level, urgency repair level, flight cancellation risk, and duration to repair.",
        height=100,
    )
    st.divider()
    st.caption(f"**Endpoint:** {AZURE_OPENAI_ENDPOINT or '⚠️ not set'}")
    st.caption(f"**Deployment:** {MODEL_DEPLOYMENT or '⚠️ not set'}")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_response_id = None
        st.rerun()

# ---------------- Main ----------------
st.title("Aircraft Maintenance Chatbot")


if not AZURE_OPENAI_ENDPOINT or not MODEL_DEPLOYMENT:
    st.warning(
        "Set `AZURE_OPENAI_ENDPOINT` and `MODEL_DEPLOYMENT` in your `.env` file "
        "before chatting.",
        icon="⚠️",
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Enter a prompt...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                client, _credential = get_client()
                response = run_async(
                    get_response(
                        client,
                        prompt,
                        st.session_state.last_response_id,
                        instructions,
                    )
                )
                assistant_text = response.output_text
                st.session_state.last_response_id = response.id
            except Exception as ex:
                assistant_text = f"⚠️ Error: {ex}"

        st.markdown(assistant_text)

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})