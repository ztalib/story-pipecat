"""
Test scoring system for semantic similarity between expected and actual results.
"""
import difflib
from typing import Any


def calculate_similarity_score(expected: Any, actual: Any) -> float:
    """
    Calculate similarity score between expected and actual values.
    Returns a score between 0.0 and 1.0.
    """
    if expected == actual:
        return 1.0

    if not isinstance(expected, type(actual)):
        return 0.0

    if isinstance(expected, str) and isinstance(actual, str):
        return calculate_string_similarity(expected, actual)

    if isinstance(expected, list) and isinstance(actual, list):
        return calculate_list_similarity(expected, actual)

    if isinstance(expected, dict) and isinstance(actual, dict):
        return calculate_dict_similarity(expected, actual)

    return 0.0


def calculate_string_similarity(expected: str, actual: str) -> float:
    """Calculate similarity between two strings using sequence matching."""
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0

    # Use difflib for sequence similarity
    matcher = difflib.SequenceMatcher(None, expected.lower(), actual.lower())
    return matcher.ratio()


def calculate_list_similarity(expected: list, actual: list) -> float:
    """Calculate similarity between two lists."""
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0

    # For lists, we'll check how many items have good matches
    total_items = max(len(expected), len(actual))
    matched_score = 0.0

    # Try to match each expected item with the best actual item
    for exp_item in expected:
        best_match = 0.0
        for act_item in actual:
            similarity = calculate_similarity_score(exp_item, act_item)
            best_match = max(best_match, similarity)
        matched_score += best_match

    # Also check items in actual that weren't in expected (penalty for extras)
    extra_penalty = max(0, len(actual) - len(expected)) * 0.1

    return max(0.0, (matched_score / total_items) - extra_penalty)


def calculate_dict_similarity(expected: dict, actual: dict) -> float:
    """Calculate similarity between two dictionaries."""
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0

    all_keys = set(expected.keys()) | set(actual.keys())
    if not all_keys:
        return 1.0

    total_score = 0.0
    for key in all_keys:
        if key in expected and key in actual:
            # Both have the key - compare values
            key_score = calculate_similarity_score(expected[key], actual[key])
        elif key in expected:
            # Missing from actual - penalty
            key_score = 0.0
        else:
            # Extra in actual - small penalty
            key_score = 0.8  # Not as bad as missing

        total_score += key_score

    return total_score / len(all_keys)


def score_test_result(expected_state: dict, actual_state: dict) -> dict[str, Any]:
    """
    Score a test result and return detailed scoring information.

    Returns:
        {
            "overall_score": 0.85,
            "pass_exact": False,
            "pass_80": True,
            "breakdown": {
                "extracted_elements": 0.85,
                "people": 0.90,
                "key_events": 0.80,
                ...
            }
        }
    """
    # Calculate overall similarity
    overall_score = calculate_similarity_score(expected_state, actual_state)

    # Calculate breakdown scores
    breakdown = {}
    if "extracted_elements" in expected_state:
        extracted_expected = expected_state["extracted_elements"]
        extracted_actual = actual_state.get("extracted_elements", {})

        breakdown["extracted_elements"] = calculate_similarity_score(
            extracted_expected, extracted_actual
        )

        # Individual element scores
        for element_type in extracted_expected:
            if element_type in extracted_actual:
                breakdown[element_type] = calculate_similarity_score(
                    extracted_expected[element_type],
                    extracted_actual[element_type]
                )
            else:
                breakdown[element_type] = 0.0

    return {
        "overall_score": round(overall_score, 3),
        "pass_exact": overall_score >= 1.0,
        "pass_80": overall_score >= 0.8,
        "pass_70": overall_score >= 0.7,
        "pass_60": overall_score >= 0.6,
        "breakdown": {k: round(v, 3) for k, v in breakdown.items()}
    }


def format_score_report(scenario_name: str, score_result: dict[str, Any]) -> str:
    """Format a score result into a readable report."""
    score = score_result["overall_score"]

    # Determine pass status
    if score_result["pass_exact"]:
        status = "✅ EXACT PASS"
    elif score_result["pass_80"]:
        status = "🟢 80%+ PASS"
    elif score_result["pass_70"]:
        status = "🟡 70%+ PARTIAL"
    elif score_result["pass_60"]:
        status = "🟠 60%+ PARTIAL"
    else:
        status = "❌ FAIL"

    report = f"{status} {scenario_name}: {score:.1%}"

    # Add breakdown if available
    if score_result["breakdown"]:
        details = []
        for element, element_score in score_result["breakdown"].items():
            if element != "extracted_elements":
                details.append(f"{element}:{element_score:.1%}")

        if details:
            report += f" ({', '.join(details)})"

    return report
