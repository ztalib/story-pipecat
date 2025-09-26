# Test Scenarios Overview

This directory contains numbered test scenarios to comprehensively test the story-capturing bot's ability to extract different types of story elements.

## Single-Turn Tests (Basic Story Elements)

1. **`1_simple_story_capture.yaml`** - Basic person capture (mother Maria as cook)
2. **`2_location_capture.yaml`** - Location/city capture (Chicago)
3. **`3_marriage_capture.yaml`** - Spouse relationships and marriage events
4. **`4_neighbor_capture.yaml`** - Neighbor relationships and descriptions
5. **`5_friend_capture.yaml`** - Friendship relationships and shared activities
6. **`6_family_event_capture.yaml`** - Significant family events with emotions
7. **`7_emotional_memory.yaml`** - Emotional moments and feelings

## Multi-Turn Tests (Complex Conversations)

8. **`8_multi_turn_family_story.yaml`** - Building family story across multiple turns
9. **`9_multi_turn_neighborhood.yaml`** - Community and neighborhood stories
10. **`10_multi_turn_wedding_story.yaml`** - Wedding story with multiple participants

## Advanced Tests

11. **`11_complex_relationships.yaml`** - Multiple family relationships in one story

## Test Coverage

These tests verify the bot can capture:
- **People**: Names, relationships, descriptions
- **Locations**: Cities, specific places, addresses  
- **Events**: Key moments, activities, celebrations
- **Emotions**: Feelings, reactions, emotional states
- **Time**: Specific dates, periods, seasons
- **Relationships**: Family, friends, neighbors, spouses
- **Multi-turn context**: Building stories across conversation turns

## Running Tests

Run all tests with:
```bash
./run_tests.sh
```

Individual test results are saved in `test_runs/[timestamp]/[scenario_name]/`
