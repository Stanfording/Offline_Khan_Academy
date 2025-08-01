import google.generativeai as genai
from google.generativeai import types
import json
import logging
import os # To read environment variables

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Gemini API Configuration ---
# Get API key from environment variable
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set your Gemini API key.")




genai.configure(api_key=API_KEY)

# Define the models to use
MODEL_NAME = "gemini-2.5-flash-lite" # Using 1.5 Flash as a robust, fast option. Change if 2.5 Flash-Lite is a specific name for you.
SUMMARIZER_MODEL = "gemini-2.5-flash-lite" # Often good to use the same model or a smaller one for summarization

# --- CONTEXT MANAGEMENT CONFIGURATION (Same as before, adjusted for Gemini) ---
NUM_RECENT_CHATS_TO_KEEP = 3 
NUM_CHATS_BEFORE_SUMMARIZATION = 6 

# Initialize Gemini generative models
main_model = genai.GenerativeModel(MODEL_NAME)
summarizer_model = genai.GenerativeModel(SUMMARIZER_MODEL)

# --- TUTOR SYSTEM PROMPT ---
with open(os.path.join(os.path.dirname(__file__), "prompt3_1.txt"), "r", encoding="utf-8") as f:
    TUTOR_SYSTEM_PROMPT = f.read()



# Rest of tutor.py remains the same (summarization logic and ask_tutor function)
# Note: The ask_tutor function does not need changes, as it already passes the
# full `current_status_dictionary` (which now contains new fields) to the LLM.
# The summarization logic is also robust enough to handle the new fields.

def _summarize_conversation(messages_to_summarize):
    """
    Calls Gemini to summarize a list of messages.
    This function needs to parse the stringified JSON messages to extract content for summarization.
    """
    extracted_content = []
    for msg in messages_to_summarize:
        if msg["role"] == "user":
            try:
                user_data = json.loads(msg["content"])
                if "user_action" in user_data and "parameters" in user_data["user_action"]:
                    # More robust extraction for user actions
                    action_params = user_data["user_action"]["parameters"]
                    command = user_data["user_action"]["command"]
                    if command == "topic_query" and "query" in action_params:
                        extracted_content.append(f"User wants to learn about: {action_params['query']}")
                    elif command == "evaluate_paraphrase" and "user_input" in action_params:
                        extracted_content.append(f"User paraphrased: {action_params['user_input']}")
                    elif command == "evaluate_answer" and "user_answer" in action_params:
                        extracted_content.append(f"User answered Q '{action_params.get('question_id', 'unknown')}': {action_params['user_answer']}")
                    elif command == "handle_user_question" and "user_question_value" in action_params:
                        extracted_content.append(f"User asked general question: {action_params['user_question_value']}")
                    # Add specific handlers for persistent button commands if their user_action parameters are crucial for summary
                    elif command in ["reveal_status", "give_feedback", "next_lesson", "skip_practice"]:
                        extracted_content.append(f"User clicked '{command}' button.")
                    else: # Generic catch-all for other commands with parameters
                        extracted_content.append(f"User action '{command}' with params: {json.dumps(action_params)}")
                elif "status_dictionary" in user_data: # If only status dictionary was sent (e.g., for init or context)
                     extracted_content.append(f"Initial state/Status update: {json.dumps(user_data['status_dictionary'])}")
            except json.JSONDecodeError:
                extracted_content.append(f"User: {msg['content']}") # Fallback for non-JSON user messages
        elif msg["role"] == "assistant":
            try:
                assistant_response = json.loads(msg["content"])
                # Extract content from assistant's actions
                for action in assistant_response.get("actions", []):
                    if action["command"] == "ui_display_notes":
                        extracted_content.append(f"Assistant Note: {action['parameters']['content']}")
                    elif "question_text" in action.get("parameters", {}):
                        extracted_content.append(f"Assistant Question: {action['parameters']['question_text']}")
                    elif action["command"] == "update_status" and "updates" in action["parameters"]:
                        if "current_lesson_stage" in action["parameters"]["updates"]:
                            extracted_content.append(f"Lesson stage changed to: {action['parameters']['updates']['current_lesson_stage']}")
                        if "current_lesson_progress" in action["parameters"]["updates"]:
                            extracted_content.append(f"Lesson progress changed to: {action['parameters']['updates']['current_lesson_progress']}%")
                        # Add other key status updates that are important for summary
            except json.JSONDecodeError:
                extracted_content.append(f"Assistant: {msg['content']}")

    summarization_prompt_content = "\n".join(extracted_content)
    if not summarization_prompt_content.strip():
        logging.warning("No content extracted for summarization. Skipping API call.")
        return "No relevant previous conversation to summarize."

    summarization_messages_payload = [
        {"role": "user", "parts": [
            "You are a helpful assistant. Summarize the following conversation history for context for a new turn. Focus on key facts, decisions, and remaining open questions, specifically related to the learning topic and user's progress. Keep it concise, but ensure all critical information is retained for future interaction. Do not add any new information. Format as plain text.",
            summarization_prompt_content
        ]}
    ]

    try:
        logging.info("   (Calling Gemini for summarization...)")
        response = summarizer_model.generate_content(
            summarization_messages_payload,
            safety_settings={'HARM_CATEGORY_HARASSMENT':'BLOCK_NONE', 'HARM_CATEGORY_HATE_SPEECH':'BLOCK_NONE', 'HARM_CATEGORY_SEXUALLY_EXPLICIT':'BLOCK_NONE', 'HARM_CATEGORY_DANGEROUS_CONTENT':'BLOCK_NONE'}, # Adjust as needed
            generation_config=types.GenerationConfig(
                temperature=0.2, # Low temperature for factual summary
                candidate_count=1,
            ),
        )
        
        summary_content = response.text
        logging.info(f"   (Summarization complete. Summary length: {len(summary_content)} characters)")
        return summary_content
    except Exception as e: # Catch broader exceptions for API calls
        logging.error(f"Error summarizing conversation with Gemini: {e}")
        return "Previous conversation context lost due to summarization error."


def ask_tutor(user_action_data, current_status_dictionary, messages_history=None):
    """
    Asks the Gemini model, managing conversational history with JSON.
    Returns the full JSON response from the LLM and the updated full messages_history.
    """
    logging.info(f"-> Ask_tutor received: user_action={user_action_data.get('command')}, status_dict_keys={current_status_dictionary.keys()}")
    
    # 1. Ensure messages_history is correctly initialized with the system prompt
    if messages_history is None or len(messages_history) == 0 or \
       not (messages_history[0].get("role") == "system" and messages_history[0].get("content") == TUTOR_SYSTEM_PROMPT):
        logging.info("   (Initializing full messages_history with system prompt for internal use)")
        messages_history = [{
            "role": "system",
            "content": TUTOR_SYSTEM_PROMPT
        }]
    else:
        logging.info(f"   (Using existing full messages_history, length: {len(messages_history)})")
        
    # 2. Append the current user input (user_action and status_dictionary) to the *full* messages_history
    # This input is stringified JSON for the LLM to parse within its context.
    user_input_content = json.dumps({
        "user_action": user_action_data,
        "status_dictionary": current_status_dictionary # Send the full status dictionary
    })
    messages_history.append({
        "role": "user",
        "content": user_input_content
    })
    logging.info(f"   (Appended current user input to full history. New length: {len(messages_history)})")

    # --- 3. Constructing 'messages_to_send' for the current Gemini API call based on memory strategy ---
    current_payload_messages_for_gemini = []

    # Identify the actual conversational turns (excluding the initial system prompt)
    conversational_messages = messages_history[1:] # All messages after the initial system prompt

    # Check if we need to summarize
    # (len(conversational_messages)) is number of user-assistant messages
    # We want to summarize older messages, but keep NUM_RECENT_CHATS_TO_KEEP user-assistant pairs
    # plus the *current* user message. This means 2 * NUM_RECENT_CHATS_TO_KEEP + 1 messages are recent.
    # Total messages in conversational_messages is (user_message + assistant_message) * N + current_user_message
    if (len(conversational_messages)) > (2 * NUM_CHATS_BEFORE_SUMMARIZATION):
        # Calculate how many messages to remove from the start for summarization
        # e.g., if we keep 3 pairs (6 messages) + current (1 message) = 7 recent messages.
        # If total is 10, summarize 10-7=3 messages.
        num_messages_to_keep_recent = (2 * NUM_RECENT_CHATS_TO_KEEP) + 1
        num_messages_to_summarize = len(conversational_messages) - num_messages_to_keep_recent
        
        if num_messages_to_summarize > 0:
            messages_to_summarize_raw = conversational_messages[:num_messages_to_summarize]
            logging.info(f"   (Detected need for summarization. Summarizing {len(messages_to_summarize_raw)} messages.)")
            summary_text = _summarize_conversation(messages_to_summarize_raw)
            
            # Add the summary as a user message, then immediately have an assistant message
            current_payload_messages_for_gemini.append({
                "role": "user",
                "parts": [f"Here is a summary of our past conversation to help you remember context: {summary_text}"]
            })
            current_payload_messages_for_gemini.append({
                "role": "model",
                "parts": ["Okay, I have reviewed the summary."]
            })
            logging.info("   (Added summarized context to payload)")
        else:
            logging.info("   (Not enough old messages to summarize, skipping summarization for now.)")
        
        # Add the recent chats (user/assistant pairs) and the current user question
        recent_chats_and_current_question = conversational_messages[num_messages_to_summarize:]
        for msg in recent_chats_and_current_question:
             # Gemini expects 'model' for assistant, 'user' for user
            role = "model" if msg["role"] == "assistant" else "user"
            current_payload_messages_for_gemini.append({"role": role, "parts": [msg["content"]]})
        
        logging.info(f"   (Added {len(recent_chats_and_current_question)} recent messages and current question to payload)")

    else:
        # History is not long enough to summarize, send all conversational turns and the current question
        logging.info("   (History is short, sending full conversational history to LLM.)")
        for msg in conversational_messages:
            role = "model" if msg["role"] == "assistant" else "user"
            current_payload_messages_for_gemini.append({"role": role, "parts": [msg["content"]]})

    # Add the overall TUTOR_SYSTEM_PROMPT as the first part of the first user message
    final_gemini_messages = [
        {"role": "user", "parts": [TUTOR_SYSTEM_PROMPT] + current_payload_messages_for_gemini[0]["parts"]},
    ]
    # Append the rest of the messages from the curated payload
    final_gemini_messages.extend(current_payload_messages_for_gemini[1:])

    logging.info(f"   (Payload for Gemini has {len(final_gemini_messages)} messages.)")

    llm_response_json = {}
    try:
        response = main_model.generate_content(
            final_gemini_messages,
            safety_settings={'HARM_CATEGORY_HARASSMENT':'BLOCK_NONE', 'HARM_CATEGORY_HATE_SPEECH':'BLOCK_NONE', 'HARM_CATEGORY_SEXUALLY_EXPLICIT':'BLOCK_NONE', 'HARM_CATEGORY_DANGEROUS_CONTENT':'BLOCK_NONE'}, # Adjust as needed
            generation_config=types.GenerationConfig(
                temperature=0.7, # Adjust creativity as needed
                candidate_count=1,
            ),
        )
        
        full_assistant_response_content = response.text
        logging.info(f"   (Gemini response received. Attempting to parse JSON.)")

        # After getting the full response, attempt to parse it as JSON
        try:
            llm_response_json = json.loads(full_assistant_response_content)
            logging.info(f"   (Successfully parsed LLM's JSON response.)")
        except json.JSONDecodeError as e:
            logging.error(f"   (ERROR: LLM did not output valid JSON: {e}. Raw content: {full_assistant_response_content[:500]}...)")
            # Fallback error response
            return {"actions": [{"command": "ui_display_notes", "parameters": {"content": f"Mr. Delight had a parsing error: {str(e)}. Please try again!"}}], "raw_output": full_assistant_response_content}, messages_history
        
        # Append the assistant's full accumulated (and parsed) JSON response to the *full* messages_history
        messages_history.append({
            "role": "assistant",
            "content": full_assistant_response_content 
        })
        logging.info(f"   (Appended assistant's full response to full history. Final full history length: {len(messages_history)})")
        
        # Return the parsed JSON response and the updated full messages_history
        return llm_response_json, messages_history

    except Exception as e: # Catch broader exceptions for API calls
        logging.error(f"   (ERROR: Gemini API call failed: {e})")
        return {"actions": [{"command": "ui_display_notes", "parameters": {"content": f"Mr. Delight is offline: {str(e)}. Please refresh and try again."}}]}, messages_history

# Test for tutor.py (will be simpler now as it's not streaming content directly)
if __name__ == "__main__":
    logging.info("--- Starting Tutor.py Test Run for JSON Output (Gemini) ---")
    
    current_messages_history = None
    current_status = {} # Initial empty status

    # Simulate an 'init' command
    logging.info("\n--- Simulate 'init' command ---")
    user_action = {"command": "init", "parameters": {"user_name": "TestLearner"}}
    llm_response, current_messages_history = ask_tutor(user_action, current_status, current_messages_history)
    logging.info(f"LLM Response (init): {json.dumps(llm_response, indent=2)}")
    
    # After init, we need to apply the status updates from the LLM's response
    # This logic is what app.py will do.
    for action in llm_response.get("actions", []):
        if action["command"] == "update_status":
            for key, value_str in action["parameters"]["updates"].items():
                if isinstance(value_str, str) and (value_str.startswith('+') or value_str.startswith('-')):
                    try:
                        delta = float(value_str)
                        if '.' in key:
                            parts = key.split('.')
                            temp_dict = current_status
                            for i, part in enumerate(parts):
                                if i == len(parts) - 1:
                                    temp_dict[part] = temp_dict.get(part, 0.0) + delta
                                else:
                                    temp_dict = temp_dict.setdefault(part, {})
                        else:
                            current_status[key] = current_status.get(key, 0.0) + delta
                    except ValueError:
                        logging.warning(f"Could not parse relative update value: {value_str}")
                else:
                    try:
                        val = json.loads(value_str) if isinstance(value_str, str) and (value_str.startswith('{') or value_str.startswith('[')) else value_str
                        if '.' in key:
                            parts = key.split('.')
                            temp_dict = current_status
                            for i, part in enumerate(parts):
                                if i == len(parts) - 1:
                                    temp_dict[part] = val
                                else:
                                    temp_dict = temp_dict.setdefault(part, {})
                        else:
                            current_status[key] = val
                    except json.JSONDecodeError:
                        current_status[key] = value_str

    logging.info(f"Updated Status after init: {json.dumps(current_status, indent=2)}")


    # Simulate a 'generate_course_plan' command
    logging.info("\n--- Simulate 'generate_course_plan' command ---")
    user_action = {"command": "generate_course_plan", "parameters": {"query": "Math for 5th grade"}}
    llm_response, current_messages_history = ask_tutor(user_action, current_status, current_messages_history)
    logging.info(f"LLM Response (generate_course_plan): {json.dumps(llm_response, indent=2)}")

    logging.info("--- Tutor.py Test Run Completed ---")