import json
from typing import Dict, Any

from pipecat.services.llm_service import FunctionCallParams
from pipecat.adapters.schemas.function_schema import FunctionSchema

# NEW: Base class for type hinting
class BaseHandler:
    NAME: str

    def get_schema(self) -> FunctionSchema:
        raise NotImplementedError

    async def handle_request(self, params: FunctionCallParams):
        raise NotImplementedError

# A simple in-memory "database" to hold story elements for each conversation.
# The key will be the pipeline task's unique ID.
# story_memory_db: Dict[str, Dict[str, Any]] = {}

# NEW CLASS to hold the state for one conversation
class ConversationMemory:
    def __init__(self):
        self.extracted_elements: Dict[str, Any] = {}
        self.conversation_state: str = "opening"

    def to_dict(self):
        return {
            "extracted_elements": self.extracted_elements,
            "conversation_state": self.conversation_state
        }

class StoryCaptureHandler(BaseHandler):
    NAME = "capture_story_elements"

    def __init__(self, data_filepath: str, memory: ConversationMemory):
        self.data_filepath = data_filepath
        self.memory = memory

    @staticmethod
    def get_schema() -> FunctionSchema:
        return FunctionSchema(
            name=StoryCaptureHandler.NAME,
            description="Extract and save important, related story elements like people, places, events, and emotions.",
            properties={
                "people": {
                    "type": "array",
                    "items": { "type": "object", "properties": {
                        "name": {"type": "string"},
                        "relationship": {"type": "string"},
                        "description": {"type": "string"}
                    }, "required": ["name", "relationship"]}
                },
                "key_events": {
                    "type": "array",
                    "items": { "type": "object", "properties": {
                        "event_name": {"type": "string"},
                        "participants": {"type": "array", "items": {"type": "string"}}
                    }, "required": ["event_name"]}
                },
                "setting": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}, "time_period": {"type": "string"}}
                },
                "emotions": {"type": "array", "items": {"type": "string"}}
            },
            required=[]
        )

    async def handle_request(self, params: FunctionCallParams):
        """This handler executes the story capture tool and saves the data."""
        
        for key, value in params.arguments.items():
            if value:
                if key not in self.memory.extracted_elements:
                    self.memory.extracted_elements[key] = []
                if isinstance(value, list):
                    self.memory.extracted_elements[key].extend(value)
                else:
                    self.memory.extracted_elements[key].append(value)

        print(f"Captured elements: {params.arguments}")
        
        with open(self.data_filepath, "w") as f:
            json.dump(self.memory.to_dict(), f, indent=2)

        await params.result_callback({"status": "success", "message": "Elements captured."})


class StoryFlowHandler(BaseHandler):
    NAME = "assess_story_flow"

    def __init__(self, data_filepath: str, memory: ConversationMemory):
        self.data_filepath = data_filepath
        self.memory = memory

    @staticmethod
    def get_schema() -> FunctionSchema:
        return FunctionSchema(
            name=StoryFlowHandler.NAME,
            description="Evaluate the story's completeness and determine next conversational move.",
            properties={
                "story_phase": {"type": "string", "enum": ["setup", "development", "climax", "resolution", "complete"]},
                "emotional_engagement": {"type": "string", "enum": ["high", "medium", "low", "overwhelmed"]},
                "next_question_type": {"type": "string", "enum": ["detail", "emotion", "context", "transition", "wrap_up"]},
                "reasoning": {"type": "string"}
            },
            required=["story_phase", "next_question_type", "reasoning"]
        )

    async def handle_request(self, params: FunctionCallParams):
        """This handler executes the story flow assessment tool and saves the state."""
        
        self.memory.conversation_state = params.arguments.get("story_phase", self.memory.conversation_state)

        print(f"Story flow assessment: {params.arguments}")

        with open(self.data_filepath, "w") as f:
            json.dump(self.memory.to_dict(), f, indent=2)

        await params.result_callback({"status": "success", "message": "Flow assessed."})
