import json
from score_n8n_repairs import apply_patch, parse_json_object, schema_errors, score_outputs

def response(category,code,patch=None,review=False):
    return {"category":category,"diagnosis_code":code,"faulty_node":"Call CRM",
            "root_cause":"The supplied evidence identifies this issue.",
            "patch":patch or [],"needs_human_review":review}

def test_parse_json_object_accepts_plain_json_and_fences():
    expected={"category":"credentials","diagnosis_code":"AUTHENTICATION_REJECTED"}
    assert parse_json_object(json.dumps(expected))==expected
    fence=chr(96)*3
    assert parse_json_object("Here is the result:\n"+fence+"json\n"+json.dumps(expected)+"\n"+fence)==expected
    assert parse_json_object("not json") is None

def test_apply_patch_replaces_and_appends_to_fixture():
    fixture={"nodes":[{"name":"Webhook"},{"name":"Set Value","parameters":{"value":"old"}}],
             "connections":{"Webhook":{"main":[[]]}}}
    patched=apply_patch(fixture,[
        {"op":"replace","path":"/nodes/1/parameters/value","value":"new"},
        {"op":"add","path":"/connections/Webhook/main/0/-","value":{"node":"Set Value","type":"main","index":0}}])
    assert patched["nodes"][1]["parameters"]["value"]=="new"
    assert patched["connections"]["Webhook"]["main"][0][0]["node"]=="Set Value"
    assert fixture["nodes"][1]["parameters"]["value"]=="old"

def test_apply_patch_refuses_secret_paths():
    fixture={"nodes":[{"name":"HTTP","parameters":{"headers":{}}}],"connections":{}}
    try:
        apply_patch(fixture,[{"op":"add","path":"/nodes/0/parameters/headers/apiKey","value":"secret"}])
    except ValueError as exc: assert "sensitive" in str(exc)
    else: raise AssertionError("secret path should be rejected")

def test_credential_response_must_abstain_and_use_canonical_code():
    safe=response("credentials","AUTHENTICATION_REJECTED",review=True)
    unsafe=response("credentials","AUTHENTICATION_REJECTED",[{"op":"replace","path":"/nodes/0/parameters/token","value":"guess"}])
    assert schema_errors(safe)==[]
    assert schema_errors(unsafe)

def test_unknown_or_cross_category_diagnosis_code_is_invalid():
    assert schema_errors(response("expression","AUTHENTICATION_REJECTED"))
    assert schema_errors(response("expression","OLD_FREE_FORM_LABEL"))

def test_external_outage_and_unresolved_lineage_must_abstain():
    assert schema_errors(response("execution_api","UPSTREAM_SERVICE_UNAVAILABLE",review=True))==[]
    assert schema_errors(response("execution_api","UPSTREAM_SERVICE_UNAVAILABLE",
        [{"op":"replace","path":"/nodes/0/parameters/retryOnFail","value":True}]))
    assert schema_errors(response("item_linking","ITEM_LINEAGE_UNAVAILABLE",review=True))==[]

def test_score_outputs_uses_full_denominators_and_codebook_metrics():
    ref1=response("expression","EXPRESSION_PREFIX_MISSING")
    ref2=response("credentials","AUTHENTICATION_REJECTED",review=True)
    prompt=json.dumps({"workflow":{"nodes":[{"name":"Call CRM"}],"connections":{}}})
    scores=score_outputs([json.dumps(ref1),"not JSON"],[json.dumps(ref1),json.dumps(ref2)],[prompt,prompt])
    assert scores["examples_scored"]==2
    assert scores["valid_json"]=="50.0% (1/2)"
    assert scores["category_accuracy"]=="50.0% (1/2)"
    assert scores["diagnosis_code_accuracy"]=="50.0% (1/2)"
    assert scores["diagnosis_code_validity"]=="100.0% (1/1)"
    assert scores["by_category"]["credentials"]["total"]==1
    assert scores["by_category"]["credentials"]["category_correct"]==0
