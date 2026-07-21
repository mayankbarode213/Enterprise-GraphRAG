"""
test_cypher_guardrail.py — Unit tests for CypherGuardrail security module
"""
import pytest
from app.graph.cypher_guardrail import CypherGuardrail


def test_valid_match_query_passes():
    query = "MATCH (s:Supplier)-[:SUPPLIED_BATCH]->(b:Batch) RETURN s, b"
    is_valid, sanitized, err = CypherGuardrail.validate_and_sanitize(query)
    assert is_valid is True
    assert sanitized == query
    assert err == ""


def test_forbidden_delete_query_blocked():
    query = "MATCH (n:Machine) DELETE n"
    is_valid, sanitized, err = CypherGuardrail.validate_and_sanitize(query)
    assert is_valid is False
    assert "Forbidden mutation operation" in err


def test_forbidden_detach_delete_blocked():
    query = "MATCH (n) DETACH DELETE n"
    is_valid, sanitized, err = CypherGuardrail.validate_and_sanitize(query)
    assert is_valid is False
    assert "Forbidden mutation operation" in err


def test_forbidden_create_blocked():
    query = "CREATE (n:Machine {name: 'Hacked'})"
    is_valid, sanitized, err = CypherGuardrail.validate_and_sanitize(query)
    assert is_valid is False
    assert "Forbidden mutation operation" in err


def test_invalid_syntax_without_match_blocked():
    query = "SELECT * FROM Users"
    is_valid, sanitized, err = CypherGuardrail.validate_and_sanitize(query)
    assert is_valid is False
    assert "does not contain valid Cypher read clauses" in err


def test_multi_where_sanitization():
    query = "MATCH (m:Machine) WHERE toLower(m.name) = 'apex' WHERE toLower(m.id) = 'rw101' RETURN m"
    is_valid, sanitized, err = CypherGuardrail.validate_and_sanitize(query)
    assert is_valid is True
    assert "WHERE toLower(m.name) = 'apex' AND toLower(m.id) = 'rw101'" in sanitized

