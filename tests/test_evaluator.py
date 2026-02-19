"""
Tests for Evaluator functionality.
"""

import pytest
from unittest.mock import Mock, patch
from rag_chatbot.evaluator import (
    RAGEvaluator,
    EvaluationMetrics,
    EvaluationResult,
    evaluate_batch,
    summarize_evaluations
)


def test_evaluation_metrics_initialization():
    """Test EvaluationMetrics initialization and overall score calculation."""
    metrics = EvaluationMetrics(
        faithfulness=0.8,
        answer_relevance=0.9,
        context_relevance=0.7,
        factual_correctness=0.85
    )
    
    assert metrics.faithfulness == 0.8
    assert metrics.answer_relevance == 0.9
    assert metrics.context_relevance == 0.7
    assert metrics.factual_correctness == 0.85
    
    # Overall score should be average of all metrics
    expected_overall = (0.8 + 0.9 + 0.7 + 0.85) / 4
    assert abs(metrics.overall_score - expected_overall) < 0.01


def test_evaluation_metrics_without_factual():
    """Test EvaluationMetrics without factual correctness."""
    metrics = EvaluationMetrics(
        faithfulness=0.8,
        answer_relevance=0.9,
        context_relevance=0.7
    )
    
    # Overall score should be average of three metrics
    expected_overall = (0.8 + 0.9 + 0.7) / 3
    assert abs(metrics.overall_score - expected_overall) < 0.01


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI response."""
    mock_response = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = "SCORE: 0.85\nREASONING: This is a test reasoning."
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    return mock_response


def test_parse_evaluation_response():
    """Test parsing of evaluation response."""
    evaluator = RAGEvaluator(api_key="test-key")
    
    content = "SCORE: 0.75\nREASONING: The answer is well grounded in the context."
    score, reasoning = evaluator._parse_evaluation_response(content)
    
    assert score == 0.75
    assert "well grounded" in reasoning


def test_parse_evaluation_response_invalid_score():
    """Test parsing with invalid score."""
    evaluator = RAGEvaluator(api_key="test-key")
    
    content = "SCORE: invalid\nREASONING: Some reasoning."
    score, reasoning = evaluator._parse_evaluation_response(content)
    
    # Should default to 0.5
    assert score == 0.5
    assert reasoning == "Some reasoning."


def test_parse_evaluation_response_clamping():
    """Test that scores are clamped to [0, 1]."""
    evaluator = RAGEvaluator(api_key="test-key")
    
    # Test score > 1
    content = "SCORE: 1.5\nREASONING: Test"
    score, _ = evaluator._parse_evaluation_response(content)
    assert score == 1.0
    
    # Test score < 0
    content = "SCORE: -0.5\nREASONING: Test"
    score, _ = evaluator._parse_evaluation_response(content)
    assert score == 0.0


def test_evaluate_batch_empty():
    """Test batch evaluation with empty list."""
    evaluator = RAGEvaluator(api_key="test-key")
    results = evaluate_batch(evaluator, [])
    assert len(results) == 0


def test_summarize_evaluations_empty():
    """Test summarization with empty results."""
    summary = summarize_evaluations([])
    assert summary == {}


def test_summarize_evaluations():
    """Test summarization of evaluation results."""
    # Create mock results
    results = []
    for i in range(3):
        metrics = EvaluationMetrics(
            faithfulness=0.8 + i * 0.05,
            answer_relevance=0.7 + i * 0.1,
            context_relevance=0.75 + i * 0.05,
            factual_correctness=0.85 + i * 0.05
        )
        result = EvaluationResult(
            query=f"Query {i}",
            answer=f"Answer {i}",
            context=[f"Context {i}"],
            ground_truth=f"Truth {i}",
            metrics=metrics,
            detailed_feedback={}
        )
        results.append(result)
    
    summary = summarize_evaluations(results)
    
    assert summary["total_cases"] == 3
    assert "faithfulness" in summary
    assert "answer_relevance" in summary
    assert "context_relevance" in summary
    assert "factual_correctness" in summary
    assert "overall_score" in summary
    
    # Check that mean is calculated correctly for faithfulness
    expected_mean = (0.8 + 0.85 + 0.9) / 3
    assert abs(summary["faithfulness"]["mean"] - expected_mean) < 0.01


def test_summarize_evaluations_without_factual():
    """Test summarization when some results don't have factual correctness."""
    results = []
    for i in range(2):
        metrics = EvaluationMetrics(
            faithfulness=0.8,
            answer_relevance=0.9,
            context_relevance=0.7,
            factual_correctness=None  # No factual correctness
        )
        result = EvaluationResult(
            query=f"Query {i}",
            answer=f"Answer {i}",
            context=[f"Context {i}"],
            ground_truth=None,
            metrics=metrics,
            detailed_feedback={}
        )
        results.append(result)
    
    summary = summarize_evaluations(results)
    
    # Should not include factual_correctness in summary
    assert "factual_correctness" not in summary
