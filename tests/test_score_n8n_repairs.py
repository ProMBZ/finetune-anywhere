import json

from score_n8n_repairs import apply_patch, parse_json_object, schema_errors, score_outputs


def response(category, code, patch=None, review=False):
    return {
        "category": category,
        "diagnosis_code": code,
        "faulty_node": "Call CRM",
        "root_cause": "Evidence identifies the issue.",
        "patch": patch if patch is not None else [],
        "needs_human_review": review,
    }


def test_parse_json_object_requires_one_complete_json_object():
    expected = {"category": "credentials", "diagnosis_code": "AUTHENTICATION_REJECTED"}
    assert parse_json_object(json.dumps(expected)) == expected
    assert parse_json_object("not json") is None
    assert parse_json_object("Here is the result: " + json.dumps(expected)) is None
    assert parse_json_object(json.dumps(expected) + "\nDone.") is None
    fence = chr(96) * 3
    assert parse_json_object(fence + "json\n" + json.dumps(expected) + "\n" + fence) is None


def test_apply_patch_and_secret_guard():
    fixture = {
        "nodes": [
            {"name": "Webhook", "parameters": {}},
            {"name": "Set Value", "parameters": {"value": "old"}},
        ],
        "connections": {"Webhook": {"main": [[]]}},
    }
    patched = apply_patch(fixture, [
        {"op": "replace", "path": "/nodes/1/parameters/value", "value": "new"},
        {"op": "add", "path": "/connections/Webhook/main/0/-",
         "value": {"node": "Set Value", "type": "main", "index": 0}},
    ])
    assert patched["nodes"][1]["parameters"]["value"] == "new"
    assert fixture["nodes"][1]["parameters"]["value"] == "old"
    try:
        apply_patch(fixture, [{"op": "add", "path": "/nodes/0/parameters/token", "value": "secret"}])
    except ValueError:
        pass
    else:
        raise AssertionError("secret path should be rejected")


def test_schema_guards():
    assert schema_errors(response("credentials", "AUTHENTICATION_REJECTED", review=True)) == []
    assert schema_errors(response(
        "credentials", "AUTHENTICATION_REJECTED",
        [{"op": "replace", "path": "/nodes/0/parameters/token", "value": "guess"}],
    ))
    assert schema_errors(response("expression", "AUTHENTICATION_REJECTED"))
    assert schema_errors(response("expression", "OLD_FREE_FORM_LABEL"))
    assert schema_errors(response("execution_api", "UPSTREAM_SERVICE_UNAVAILABLE", review=True)) == []
    assert schema_errors(response(
        "execution_api", "UPSTREAM_SERVICE_UNAVAILABLE",
        [{"op": "replace", "path": "/nodes/0/parameters/retryOnFail", "value": True}],
    ))


def test_diagnosis_code_accuracy_requires_category_and_code_to_match():
    reference = response("expression", "EXPRESSION_PREFIX_MISSING")
    wrong_pair = response("data_mapping", "EXPRESSION_PREFIX_MISSING")
    prompt = json.dumps({"workflow": {"nodes": [{"name": "Call CRM", "parameters": {}}], "connections": {}}})
    scores = score_outputs(
        [json.dumps(wrong_pair)],
        [json.dumps(reference)],
        [prompt],
    )
    assert scores["category_accuracy"] == "0.0% (0/1)"
    assert scores["diagnosis_code_accuracy"] == "0.0% (0/1)"
    assert scores["diagnosis_code_raw_label_match"] == "100.0% (1/1)"
    assert scores["diagnosis_code_validity"] == "0.0% (0/1)"
    assert scores["by_diagnosis_code"]["EXPRESSION_PREFIX_MISSING"] == {"total": 1, "correct": 0}


def test_malformed_output_counts_wrong_in_full_denominators():
    reference = response("credentials", "AUTHENTICATION_REJECTED", review=True)
    good = response("expression", "EXPRESSION_PREFIX_MISSING")
    prompt = json.dumps({"workflow": {"nodes": [{"name": "Call CRM", "parameters": {}}], "connections": {}}})
    scores = score_outputs(
        [json.dumps(good), "not JSON"],
        [json.dumps(good), json.dumps(reference)],
        [prompt, prompt],
    )
    assert scores["examples_scored"] == 2
    assert scores["valid_json"] == "50.0% (1/2)"
    assert scores["malformed_json"] == "50.0% (1/2)"
    assert scores["category_accuracy"] == "50.0% (1/2)"
    assert scores["diagnosis_code_accuracy"] == "50.0% (1/2)"
    assert scores["diagnosis_code_validity"] == "50.0% (1/2)"
    assert scores["human_review_decision_accuracy"] == "50.0% (1/2)"
    assert scores["safe_abstention_recall"] == "0.0% (0/1)"
    assert scores["safe_abstention_precision"] == "n/a"


def test_patch_applicability_rejects_retry_settings_inside_parameters():
    prompt = json.dumps({
        "workflow": {
            "nodes": [
                {"name": "HTTP Request", "parameters": {"url": "https://api.example.test"}},
            ],
            "connections": {},
        }
    })
    prediction = response(
        "node_configuration",
        "TIMEOUT_TOO_SHORT",
        [{"op": "add", "path": "/nodes/0/parameters/retryOnFail", "value": True}],
    )
    scores = score_outputs(
        [json.dumps(prediction)],
        [json.dumps(prediction)],
        [prompt],
    )
    assert scores["patches_attempted"] == 1
    assert scores["patch_applicability"] == "0.0% (0/1)"


def test_schema_handles_malformed_types():
    malformed = {
        "category": [],
        "diagnosis_code": {},
        "faulty_node": "Call CRM",
        "root_cause": "Invalid",
        "patch": [{"op": "explode", "path": [], "value": 1}],
        "needs_human_review": "yes",
    }
    assert schema_errors(malformed)
