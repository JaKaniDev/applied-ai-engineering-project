#!/usr/bin/env python3
import os
import sys
from google import genai
from google.genai import types

def start_tutor_session():
    """
    Initializes a stateful chat session with embedded system instructions
    to enforce a supportive, peer-developer Socratic tutoring persona.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        print("CRITICAL ERROR: GEMINI_API_KEY not found in environment.", file=sys.stderr)
        sys.exit(1)

    # 1. Define the behavioral boundary conditions
    socratic_instruction = """
    You are a supportive, technical peer-developer serving as a Socratic Tutor. 
    Your goal is to guide the user to deep programming insights.
    
    CRITICAL RULES:
    1. NEVER provide direct code solutions, full code blocks, or direct answers.
    2. Respond by breaking concepts down and asking exactly ONE clarifying or guiding question.
    3. Keep your responses technical, concise, and focused on the immediate conceptual step.
    """

    # 2. Initialize client and configure runtime parameters
    client = genai.Client()
    config = types.GenerateContentConfig(
        system_instruction=socratic_instruction,
        temperature=0.7,  # Balanced for conversational variety and rule adherence
    )

    print("============================================================")
    print("🎓 SOCRATIC TUTOR ACTIVE: Type 'exit' or 'quit' to end session.")
    print("============================================================\n")

    # 3. Create the stateful chat manager object
    # This automatically tracks the history array behind the scenes
    chat = client.chats.create(model="gemini-2.5-flash", config=config)

    # 4. Enter the interactive REPL (Read-Eval-Print Loop)
    while True:
        try:
            user_input = input("\nYou ➜ ")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("\nClosing session. Great work today, developer!")
                break
                
            if not user_input.strip():
                continue

            print("Tutor ➜ ", end="", flush=True)

            # Send message using the chat state object
            response_stream = chat.send_message_stream(user_input)
            
            for chunk in response_stream:
                print(chunk.text, end="", flush=True)
            print()

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting cleanly.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    start_tutor_session()
