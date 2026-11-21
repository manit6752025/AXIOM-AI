from __future__ import annotations

import pytest

from axiom.core.secret_refs import (
    SecretReferenceError,
    normalize_secret_reference,
    resolve_secret_reference,
)


@pytest.mark.parametrize("name", [
    "AXIOM_PROVIDER_KEY",
    "provider_key",
    "_PRIVATE_TOKEN",
    "A",
])
def test_secret_reference_accepts_only_environment_variable_names(name: str) -> None:
    assert normalize_secret_reference(name) == name


@pytest.mark.parametrize("name", [
    "lower-case",
    "has space",
    "DOLLAR$KEY",
    "KEY\nINJECTED",
    "KEY=VALUE",
    "9INVALID",
    "A" * 129,
    123,
])
def test_secret_reference_rejects_non_names_and_injection(name: object) -> None:
    with pytest.raises(SecretReferenceError):
        normalize_secret_reference(name)


def test_empty_reference_means_no_configured_secret() -> None:
    assert normalize_secret_reference(None) is None
    assert normalize_secret_reference("") is None
    with pytest.raises(SecretReferenceError, match="No credential environment reference"):
        resolve_secret_reference(None)


def test_secret_value_is_resolved_only_from_the_named_environment_variable(monkeypatch) -> None:
    monkeypatch.setenv("AXIOM_TEST_PROVIDER_KEY", "value-never-persisted-or-rendered")
    assert resolve_secret_reference("AXIOM_TEST_PROVIDER_KEY") == "value-never-persisted-or-rendered"


@pytest.mark.parametrize("value", ["", "   ", "contains\nnewline", "x" * 513])
def test_missing_blank_or_unsafe_environment_values_fail_closed(monkeypatch, value: str) -> None:
    monkeypatch.setenv("AXIOM_TEST_PROVIDER_KEY", value)
    with pytest.raises(SecretReferenceError, match="missing or invalid"):
        resolve_secret_reference("AXIOM_TEST_PROVIDER_KEY")
