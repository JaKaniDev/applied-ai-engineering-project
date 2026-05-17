import os
import sys
import time  # FIX 1: Missing import must be present for time.sleep to work
from google import genai
from google.genai import types
# Import our freshly verified local RAG engine
from rag_engine import RAGEngine


class RAGConversationalTutor:
    def __init__(self, system_instruction_path="prompts/system_instructions/socratic_tutor_v1.md"):
        """
        Initializes the Socratic Tutor with integrated RAG context retrieval.
        """
        print("🛠️  Initializing Socratic Tutor with RAG Core...")
        # 1. Boot up the local Vector Database client
        self.rag = RAGEngine()

        # 2. Boot up the primary Gemini intelligence client
        self.ai_client = genai.Client()

        # FIX 2: Downgrade from 'pro' to 'flash' to leverage wider Free Tier API lanes
        self.model_name = "gemini-2.5-flash"

        # 3. Load our foundational Socratic rules
        self.system_instruction = self._load_system_instruction(system_instruction_path)

        # 4. Spin up our stateful conversation manager
        self.chat = self.ai_client.chats.create(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                temperature=0.2  # Kept low for deterministic engineering focus
            )
        )

    def _load_system_instruction(self, path: str) -> str:
        if not os.path.exists(path):
            print(f"⚠️  Warning: System instruction file not found at {path}. Falling back to default.")
            return "You are a helpful Socratic tutor."
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def interact(self, user_message: str):
        """
        Queries the database for background context, builds an augmented
        payload, and streams the Socratic response to the terminal.
        """
        # STEP 1: Intercept & Retrieve
        print("\n🔍 Searching local knowledge base for context...")
        relevant_chunks = self.rag.query_knowledge(user_message, n_results=1)

        # STEP 2: Augment the payload if context is found
        augmented_prompt = ""
        if relevant_chunks:
            context_block = "\n".join(relevant_chunks)
            augmented_prompt = (
                f"Use the following ground-truth context to guide your Socratic questioning:\n"
                f"<context>\n{context_block}\n</context>\n\n"
                f"User Question: {user_message}"
            )
            print("🎯 Relevant context found and injected into prompt stream.")
        else:
            augmented_prompt = user_message
            print("ℹ️  No specific context found. Processing message natively.")

        # STEP 3: Generate using immediate streaming
        print("🤖 Tutor thinking...\n")
        response_stream = self.chat.send_message_stream(augmented_prompt)

        # Flush response blocks directly to stdout for fluid terminal interaction
        for chunk in response_stream:
            sys.stdout.write(chunk.text)
            sys.stdout.flush()
        print("\n")


def main():
    # Quick execution entry point to interact with your upgraded tutor
    tutor = RAGConversationalTutor()
    print("🎓 Socratic Tutor RAG-Engine is fully active.")
    print("Type 'exit' or 'quit' to terminate the session.\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Closing tutor workspace. System state conserved.")
                break
            if not user_input.strip():
                continue

            tutor.interact(user_input)

            # Guard delay: Artificially space out sequential requests to satisfy the free tier limits
            print("\n⏳ Cooling down request limits...")
            time.sleep(15)

        except KeyboardInterrupt:
            print("\nSession safely interrupted.")
            break


if __name__ == "__main__":
    main()