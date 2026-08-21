

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure the module under test is importable regardless of CWD in CI
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODULE_NAME = "aircraft_assistant"  # change to match the actual filename (without .py)


@pytest.fixture(autouse=True)
def azure_env(monkeypatch):
    """Provide dummy env vars so the module never touches a real endpoint."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake-endpoint.example.com")
    monkeypatch.setenv("MODEL_DEPLOYMENT", "fake-model-deployment")


@pytest.fixture
def mock_azure_credential():
    """Patch DefaultAzureCredential so no real auth/network call happens."""
    with patch(f"{MODULE_NAME}.DefaultAzureCredential") as mock_cred_cls:
        instance = MagicMock()
        instance.close = AsyncMock()
        mock_cred_cls.return_value = instance
        yield mock_cred_cls, instance


@pytest.fixture
def mock_token_provider():
    with patch(f"{MODULE_NAME}.get_bearer_token_provider") as mock_provider:
        mock_provider.return_value = "fake-token-provider"
        yield mock_provider


@pytest.fixture
def mock_openai_client():
    """Patch AsyncOpenAI so response creation is fully in-memory."""
    with patch(f"{MODULE_NAME}.AsyncOpenAI") as mock_client_cls:
        instance = MagicMock()
        fake_response = MagicMock()
        fake_response.output_text = "This is a mocked assistant reply."
        fake_response.id = "resp_fake_123"
        instance.responses.create = AsyncMock(return_value=fake_response)
        mock_client_cls.return_value = instance
        yield mock_client_cls, instance


class TestModuleImport:
    def test_module_imports_without_error(self):
        """Sanity check: module loads even without real Azure/OpenAI creds."""
        __import__(MODULE_NAME)

    def test_env_vars_are_read(self, azure_env):
        importlib_reload_check = __import__(MODULE_NAME)
        assert os.getenv("AZURE_OPENAI_ENDPOINT") == "https://fake-endpoint.example.com"
        assert os.getenv("MODEL_DEPLOYMENT") == "fake-model-deployment"


class TestMainLoop:
    @pytest.mark.asyncio
    async def test_quit_exits_immediately_and_closes_credential(
        self, mock_azure_credential, mock_token_provider, mock_openai_client, monkeypatch
    ):
        """User types 'quit' right away: no API call should be made, and the
        Azure credential session should still be closed cleanly."""
        module = __import__(MODULE_NAME)

        monkeypatch.setattr("builtins.input", lambda _: "quit")

        await module.main()

        _, client_instance = mock_openai_client
        client_instance.responses.create.assert_not_awaited()

        _, cred_instance = mock_azure_credential
        cred_instance.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_single_prompt_then_quit_calls_responses_create(
        self, mock_azure_credential, mock_token_provider, mock_openai_client, monkeypatch, capsys
    ):
        """One real prompt followed by 'quit' should trigger exactly one
        call to responses.create with the expected instructions/model."""
        module = __import__(MODULE_NAME)

        inputs = iter(["What's the diagnosis for a hydraulic leak?", "quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        await module.main()

        _, client_instance = mock_openai_client
        client_instance.responses.create.assert_awaited_once()
        _, kwargs = client_instance.responses.create.call_args
        assert kwargs["model"] == "fake-model-deployment"
        assert kwargs["input"] == "What's the diagnosis for a hydraulic leak?"
        assert "Aircraft Maintenance" in kwargs["instructions"]

        captured = capsys.readouterr()
        assert "This is a mocked assistant reply." in captured.out

    @pytest.mark.asyncio
    async def test_empty_input_is_skipped(
        self, mock_azure_credential, mock_token_provider, mock_openai_client, monkeypatch, capsys
    ):
        """Blank input should print a nudge and not call the API."""
        module = __import__(MODULE_NAME)

        inputs = iter(["", "quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        await module.main()

        _, client_instance = mock_openai_client
        client_instance.responses.create.assert_not_awaited()

        captured = capsys.readouterr()
        assert "Please enter a prompt." in captured.out

    @pytest.mark.asyncio
    async def test_exception_is_caught_and_credential_still_closed(
        self, mock_azure_credential, mock_token_provider, mock_openai_client, monkeypatch, capsys
    ):
        """If responses.create raises, main() should catch it, print it,
        and still close the credential in the finally block."""
        module = __import__(MODULE_NAME)

        _, client_instance = mock_openai_client
        client_instance.responses.create.side_effect = RuntimeError("boom")

        inputs = iter(["trigger the error", "quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        await module.main()

        captured = capsys.readouterr()
        assert "boom" in captured.out

        _, cred_instance = mock_azure_credential
        cred_instance.close.assert_awaited_once()