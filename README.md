# Aircraft Maintenance Assistant

An AI-powered assistant that helps diagnose aircraft maintenance issues. It answers questions about diagnosis, risk level, urgency of repair, flight cancellation risk, and estimated repair duration, using Azure OpenAI's Responses API.

The project includes two interfaces:

- **`aircraft_assistant.py`** — a command-line chatbot.
- **`aircraft_assistant_streamlit.py`** — a Streamlit web app with chat history and configurable system instructions.

## Prerequisites

- Python 3.9+
- An Azure OpenAI resource with a deployed model
- Azure credentials available to `DefaultAzureCredential` (e.g. via `az login`)

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root with your Azure OpenAI settings:

   ```env
   AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/openai/v1"
   MODEL_DEPLOYMENT="<your-model-deployment-name>"
   ```

## Usage

### Command-line chatbot

```bash
python aircraft_assistant.py
```

Enter prompts at the terminal, and type `quit` to exit.

### Streamlit web app

```bash
streamlit run aircraft_assistant_streamlit.py
```

This opens a chat UI in your browser. Use the sidebar to view/edit the system instructions and clear the conversation.

## Notes

- Authentication uses `DefaultAzureCredential`, so make sure you're signed in to Azure (e.g. `az login`) before running either app.
- The `.env` file is git-ignored and should never be committed.
