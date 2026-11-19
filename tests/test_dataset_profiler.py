import json

from axiom.datasets.profiler import profile_jsonl


def test_profile_counts_rows_duplicates_and_types(tmp_path):
    source = tmp_path / "train.jsonl"
    source.write_text(
        '{"text":"hello","count":1}\n'
        '{"text":"hello","count":1}\n'
        '{"text":"world","count":null}\n'
        'not json\n'
        '\n',
        encoding="utf-8",
    )

    report = profile_jsonl(source)
    assert report["valid_rows"] == 3
    assert report["invalid_rows"] == 1
    assert report["empty_rows"] == 1
    assert report["duplicate_rows"] == 1
    fields = {field["name"]: field for field in report["fields"]}
    assert fields["count"]["null_count"] == 1
    assert fields["text"]["types"] == {"string": 3}


def test_profile_flags_sensitive_field_names_without_exposing_values(tmp_path):
    secret_value = "private-test-value-98765"
    source = tmp_path / "private.jsonl"
    source.write_text(json.dumps({"email": secret_value, "password_hash": "hash"}) + "\n", encoding="utf-8")

    report = profile_jsonl(source)
    rendered = json.dumps(report)
    assert report["privacy_review_recommended"] is True
    assert set(report["possibly_sensitive_fields"]) == {"email", "password_hash"}
    assert secret_value not in rendered


def test_profile_enforces_byte_and_line_limits(tmp_path):
    source = tmp_path / "large.jsonl"
    source.write_text('{"text":"' + ('x' * 256) + '"}\n{"text":"later"}\n', encoding="utf-8")

    report = profile_jsonl(source, max_bytes=32, max_rows=100, max_line_bytes=16)
    assert report["truncated"] is True
    assert "max_line_bytes" in report["truncation_reasons"]
    assert report["bytes_scanned"] <= 32
    assert report["overlong_rows"] >= 1 or report["invalid_rows"] >= 1


def test_profile_names_the_scan_limit_that_caused_truncation(tmp_path):
    source = tmp_path / "rows.jsonl"
    source.write_text('{"id":1}\n{"id":2}\n', encoding="utf-8")

    report = profile_jsonl(source, max_rows=1)

    assert report["truncated"] is True
    assert report["truncation_reasons"] == ["max_rows"]


def test_profile_rejects_invalid_limits_and_non_object_json(tmp_path):
    source = tmp_path / "mixed.jsonl"
    source.write_text('[1,2]\nnull\nNaN\n', encoding="utf-8")
    report = profile_jsonl(source)
    assert report["valid_rows"] == 0
    assert report["invalid_rows"] == 3

    try:
        profile_jsonl(source, max_rows=0)
    except ValueError as exc:
        assert "positive integers" in str(exc)
    else:
        raise AssertionError("invalid limit should be rejected")


def test_profile_reports_encoding_newlines_duplicate_keys_and_private_token_bins(tmp_path):
    source = tmp_path / "diagnostics.jsonl"
    source.write_bytes(
        b'{"id":"a","prompt":"private words here"}\r\n'
        b'{"id":"a","prompt":"other words"}\n'
        b'{"id":"b","prompt":"one"}\n'
        b'{"bad":}\n'
        b'{"prompt":"\xff"}\n'
        b'["not", "an object"]\n'
    )

    report = profile_jsonl(
        source,
        duplicate_key_fields=["id"],
        duplicate_rate_threshold=0.3,
        text_fields=["prompt"],
    )

    assert report["diagnostics"]["encoding_errors"] == 1
    assert report["diagnostics"]["malformed_json_rows"] == 1
    assert report["diagnostics"]["non_object_json_rows"] == 1
    assert report["diagnostics"]["newline_style"] == "mixed"
    assert report["duplicate_key_summary"] == {
        "fields": ["id"],
        "eligible_rows": 3,
        "duplicate_rows": 1,
        "rate": 0.333333,
        "threshold": 0.3,
        "threshold_exceeded": True,
    }
    assert report["text_token_lengths"]["prompt"]["histogram"] == {"1": 1, "2-4": 2}
    rendered = json.dumps(report)
    assert "private words here" not in rendered
    assert "other words" not in rendered


def test_profile_rejects_invalid_diagnostic_options(tmp_path):
    source = tmp_path / "valid.jsonl"
    source.write_text('{"id":1}\n', encoding="utf-8")

    for options in (
        {"duplicate_rate_threshold": 1.1},
        {"duplicate_key_fields": ["id", " id "]},
        {"text_fields": [""]},
    ):
        try:
            profile_jsonl(source, **options)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid profiler options should be rejected")


def test_profile_allows_extra_sensitive_rules_and_exact_false_positive_overrides(tmp_path):
    source = tmp_path / "privacy.jsonl"
    source.write_text('{"customer_ref":"x","email_template":"y"}\n', encoding="utf-8")

    report = profile_jsonl(
        source,
        sensitive_field_rules=["customer"],
        ignored_sensitive_fields=["email_template"],
    )

    assert report["possibly_sensitive_fields"] == ["customer_ref"]
    assert report["sensitive_field_rules"] == ["customer"]
    assert report["ignored_sensitive_fields"] == ["email_template"]


def test_pii_scan_requires_opt_in_and_returns_only_redacted_aggregates(tmp_path):
    source = tmp_path / "pii.jsonl"
    email = "private.person@example.com"
    phone = "+1 (415) 555-0123"
    card = "4111 1111 1111 1111"
    source.write_text(json.dumps({"contact": email, "notes": [phone, card]}) + "\n", encoding="utf-8")

    default_report = profile_jsonl(source)
    assert default_report["pii_scan"]["status"] == "not_run"
    assert default_report["pii_scan"]["opted_in"] is False

    report = profile_jsonl(source, scan_pii=True)
    pii_scan = report["pii_scan"]
    assert pii_scan["opted_in"] is True
    assert pii_scan["matches_by_type"] == {"email": 1, "phone_like": 1, "payment_card": 1}
    assert pii_scan["fields_by_type"] == {
        "email": ["contact"],
        "phone_like": ["notes"],
        "payment_card": ["notes"],
    }
    rendered = json.dumps(report)
    for value in (email, phone, card):
        assert value not in rendered
