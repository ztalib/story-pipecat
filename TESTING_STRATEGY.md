# Conversational AI Testing Strategy

## 1. Core Philosophy: API-First, Behavior-Driven Testing

Our testing strategy is designed to be robust, scalable, and focused on what matters most: the quality of the conversation. It is built on three core principles:

1.  **Test the Agent, Not the Plumbing:** We separate the conversational "brain" (`StoryInterviewer`) from the voice infrastructure (Pipecat, STT/TTS). We test the brain through a dedicated, text-only API endpoint, which gives us fast, reliable, and deterministic tests.

2.  **Behavior Over Implementation:** We treat the agent as a "black box." Our tests do not inspect the agent's internal code; they only look at the public API output (`response_text` and `state`). This means we can completely refactor the agent's internals, and as long as its conversational behavior remains correct, the tests will pass.

3.  **Human-Readable Scenarios:** Tests are written in simple YAML files that read like conversation scripts. This allows anyone (including non-developers) to understand, write, and contribute to the test suite. It also creates a valuable library of "living documentation" about how the agent is supposed to behave.

## 2. Testing Framework Components

*   **A `/test/chat` API Endpoint:** A text-based endpoint that returns the agent's response and a JSON object representing its current memory/state. This is the key to fast, deterministic testing.
*   **YAML "Story Scripts":** Human-readable files defining turn-by-turn conversations and the expected state after each turn.
*   **An Automated Test Runner (`pytest`):** A script that runs all YAML scenarios against the `/test/chat` API and compares the results.

### 2.1. The Test Runner: Simple & Deterministic

The test runner performs a **simple, deterministic comparison** of JSON data. It does **not** use an LLM to "judge" the response. This ensures that tests are fast, cheap, and 100% reproducible. An LLM-based evaluator is a separate tool for more advanced quality analysis, not for core functional testing.

### 2.2. Test Output: Overall Summary & Per-Test Delta

The test results provide both a high-level summary and detailed failure analysis:

1.  **Overall Summary:** A single line at the end of the run shows the final tally (e.g., `4 passed, 1 failed`).
2.  **Per-Test Delta:** For failing tests, a detailed "diff" is printed, showing exactly which scenario, turn, and data key caused the failure.

**Example Failure Output:**
```
============================== FAILURES ===============================
__ SCENARIO: tests/scenarios/multi_element_capture.yaml (Turn 1) _______

>       assert actual_state >= expected_state

E       AssertionError: The actual state from the agent did not contain the expected structure.
E       
E       --- Diff ---
E       Path: ['extracted_elements']['key_events'][0]
E       
E       - Expected (from YAML):
E         { "event_name": "built a treehouse", "participants": ["David", "user"] }
E       
E       + Actual (from API):
E         { "event_name": "built a treehouse" }
E       
E       Missing key: 'participants'
```

---

## 3. Implementation Plan

Our path to a fully testable agent involves three phases:

1.  **Phase 1: Build the Test Harness (The "Brain Surgery" API).** The first step is to create a way to interact with the agent's logic directly, bypassing the audio pipeline. We will:
    *   **Add a `/test/chat` Endpoint to `app.py`:** This new, internal API endpoint will be the core of our testing framework. It will accept a user's text utterance and the conversation's current JSON state, then execute the agent's logic and return the new, updated state. This gives us a synchronous, deterministic way to "talk" to the agent's brain.
    *   **Set up `pytest`:** We will create a `tests/` directory and configure `pytest` as our test runner.
    *   **Write a YAML Test Runner:** Inside the `tests/` directory, we'll build a Python script that knows how to find, parse, and execute the test scenarios defined in our YAML files.

2.  **Phase 2: Write "Happy Path" Scenarios.** With the harness in place, we'll create our initial library of tests.
    *   **Create a `tests/scenarios/` Directory:** All test scripts will live here.
    *   **Write Single-Utterance YAML Files:** We'll start with simple, one-turn conversations that test the agent's core data extraction capabilities, mirroring the examples below.

3.  **Phase 3: Run, Analyze, and Refine.** The final phase is to integrate testing into our development loop.
    *   **Execute Tests:** Run the full suite via a simple `pytest` command.
    *   **Analyze Failures:** Use the detailed diff output to quickly pinpoint logic errors or regressions.
    *   **Expand Coverage:** Continuously add new scenarios to cover edge cases, user corrections, and more complex conversational flows.

---

## 4. Phase 1: Happy Path Scenarios

**Goal:** To confirm core functionality works correctly under ideal conditions.

### Example 1: `simple_story_capture.yaml`
```yaml
scenario_name: "User shares a simple memory about their mother"
conversation:
  - user: "I'd like to tell you a story about my mother, Maria. She was an amazing cook."
    expect_in_response_state:
      extracted_elements:
        people:
          - name: "Maria"
            relationship: "mother"
            description: "amazing cook"
```

### Example 2: `multi_element_capture.yaml`
```yaml
scenario_name: "User provides multiple, related details across two turns"
conversation:
  - user: "My friend David and I built a treehouse back in 1995."
    expect_in_response_state:
      extracted_elements:
        people:
          - name: "David"
            relationship: "friend"
        key_events:
          - event_name: "built a treehouse"
            participants: ["David", "user"]
        setting:
          - time_period: "1995"

  - user: "It was in the big oak tree behind my childhood home."
    expect_in_response_state:
      # State should be cumulative. The test runner will verify these elements exist,
      # without requiring a perfect match of the entire state object.
      extracted_elements:
        people:
          - name: "David"
            relationship: "friend"
        key_events:
          - event_name: "built a treehouse"
            participants: ["David", "user"]
        setting:
          - time_period: "1995"
          - location: "big oak tree behind my childhood home"
```

---

## 5. Phase 2: Robustness & Resilience Scenarios

**Goal:** To ensure the agent can gracefully handle imperfect user input.

### Example 1: `user_self_correction.yaml`
```yaml
scenario_name: "User corrects a name during their utterance"
conversation:
  - user: "The main person in my story is my brother, Bill... sorry, I meant to say Will."
    expect_in_response_state:
      extracted_elements:
        people:
          - name: "Will"
            relationship: "brother"
    expect_not_in_response_state:
      extracted_elements:
        people:
          - name: "Bill"
```

### Example 2: `asr_error_simulation.yaml`
```yaml
scenario_name: "Agent handles a likely ASR error"
conversation:
  - user: "Back then we lived in a small house on the see." # ASR mistakes "sea" for "see"
    expect_not_in_response_state:
      extracted_elements:
        setting:
          - location: "see"
```

---

## 6. Future Phases
*   **Coherence Testing:** Longer conversations to test for memory and context retention.
*   **Conversational Flow Testing:** Scenarios that test handling of topic switches and interruptions.
*   **Evaluation & Replay Framework:** A system to log and replay real-world conversations to measure improvements between agent versions.
