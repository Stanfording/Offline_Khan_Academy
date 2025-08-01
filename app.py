from flask import Flask, request, jsonify, render_template, Response
from tutor import ask_tutor, TUTOR_SYSTEM_PROMPT # Import TUTOR_SYSTEM_PROMPT for init check
import json
import uuid 
import logging

app = Flask(__name__)

# Configure logging to see messages in the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dictionary to store conversation message histories AND status dictionaries per session
# Each session_id will map to a dict: {"messages": [], "status": {}}
session_data = {}

# Initial Status Dictionary structure (as per prompt)
INITIAL_STATUS_DICT = {
    "learning_confidence": 5,
    "learning_interest": 5,
    "learning_patience": 5,
    "effort_focus": 10,
    "weak_concept_spot": {},
    "current_lesson_progress": 0,
    "current_lesson_stage": "initial", # New: Default initial stage
    "current_lesson_title": "", # New: To track current lesson title
    "active_question_id": "" # New: To track the ID of the last question asked
}

@app.route('/')
def home():
    """Serves the home HTML page."""
    return render_template('home.html')

@app.route('/classroom')
def classroom():
    """Serves the classroom HTML page."""
    session_id = request.args.get('session_id', str(uuid.uuid4()))
    
    if session_id not in session_data:
        # Initialize session data with empty messages and default status
        session_data[session_id] = {
            "messages": [],
            "status": INITIAL_STATUS_DICT.copy() # Use .copy() to prevent shared references
        }
        logging.info(f"Initialized new session_id: {session_id} with default status.")
    
    logging.info(f"--- Classroom loaded for session_id: {session_id} ---")
    return render_template('classroom.html', session_id=session_id)

@app.route('/api/courses')
def get_courses():
    """Serves the course data from the JSON file."""
    try:
        with open('courses.json', 'r') as f:
            courses_data = json.load(f)
        return jsonify(courses_data)
    except FileNotFoundError:
        logging.error("courses.json not found")
        return jsonify({"error": "courses.json not found"}), 404
    except json.JSONDecodeError:
        logging.error("Error decoding courses.json")
        return jsonify({"error": "Error decoding courses.json"}), 500

@app.route('/ask', methods=['POST'])
def ask():
    req_data = request.get_json()
    user_action = req_data.get('user_action') # This is the main input from frontend
    session_id = req_data.get('session_id') 

    if not user_action or not isinstance(user_action, dict) or 'command' not in user_action:
        logging.warning("Invalid or missing 'user_action' in /ask request.")
        return jsonify({'error': 'Invalid or missing user_action'}), 400
    
    if not session_id:
        logging.warning("No session_id provided for /ask request.")
        return jsonify({'error': 'No session_id provided'}), 400

    # Retrieve current session data
    current_session = session_data.get(session_id)
    if current_session is None:
        logging.warning(f"Session ID {session_id} not found in session_data during /ask. Initializing new session.")
        current_session = {
            "messages": [],
            "status": INITIAL_STATUS_DICT.copy()
        }
        session_data[session_id] = current_session

    current_messages_history = current_session["messages"]
    current_status_dictionary = current_session["status"]

    logging.info(f"[{session_id}] - Received /ask request: command='{user_action.get('command')}', params={user_action.get('parameters')}")
    logging.info(f"[{session_id}] - Messages history length (before tutor call): {len(current_messages_history)}")
    logging.info(f"[{session_id}] - Current Status Dictionary (before tutor call): {current_status_dictionary}")

    # Call tutor.py. It will return the LLM's full JSON response and the updated message history.
    # The LLM receives the current_status_dictionary in the user_action payload
    llm_response_json, updated_messages_history = ask_tutor(
        user_action, 
        current_status_dictionary, # Pass the current status dict (which includes lesson_stage)
        current_messages_history
    )

    # Check for errors from tutor.py or LLM output
    if "error" in llm_response_json:
        logging.error(f"[{session_id}] - Error from ask_tutor: {llm_response_json['error']}")
        return jsonify({"actions": [
            {"command": "ui_display_notes", "parameters": {"content": f"Oops! Mr. Delight seems a bit stuck. Error: {llm_response_json['error']}"}}
        ], "current_status": current_status_dictionary}), 500 # Ensure status is sent even on error
    
    # Update the session's message history in our global store
    current_session["messages"] = updated_messages_history
    logging.info(f"[{session_id}] - Messages history updated. New length: {len(current_session['messages'])}")

    # --- Process and Apply Status Updates from LLM's response ---
    final_actions_for_client = []
    
    for action in llm_response_json.get("actions", []):
        if action.get("command") == "update_status":
            updates = action.get("parameters", {}).get("updates", {})
            logging.info(f"[{session_id}] - Applying status updates: {updates}")
            for key, value in updates.items(): # Changed value_str to value for clarity
                try:
                    # Handle relative updates (e.g., "+1", "-0.5")
                    if isinstance(value, str) and (value.startswith('+') or value.startswith('-')):
                        delta = float(value)
                        if '.' in key: # Handle nested keys like weak_concept_spot.Photosynthesis
                            parts = key.split('.')
                            temp_dict = current_status_dictionary
                            for i, part in enumerate(parts):
                                if i == len(parts) - 1:
                                    # Ensure the nested dict exists for accumulation
                                    if part not in temp_dict:
                                        temp_dict[part] = 0.0 # Initialize if not exists
                                    temp_dict[part] = temp_dict[part] + delta
                                else:
                                    temp_dict = temp_dict.setdefault(part, {})
                        else:
                            current_status_dictionary[key] = current_status_dictionary.get(key, 0.0) + delta
                    else:
                        # Handle direct assignments (numbers, strings, booleans, objects like {})
                        # Try to parse as JSON for objects/arrays/numbers if they aren't simple strings
                        try:
                            # Only attempt JSON load if it looks like a JSON string
                            val = json.loads(value) if isinstance(value, str) and (value.startswith('{') or value.startswith('[')) else value
                            if '.' in key: # Handle nested keys
                                parts = key.split('.')
                                temp_dict = current_status_dictionary
                                for i, part in enumerate(parts):
                                    if i == len(parts) - 1:
                                        temp_dict[part] = val
                                    else:
                                        temp_dict = temp_dict.setdefault(part, {})
                            else:
                                current_status_dictionary[key] = val
                        except json.JSONDecodeError:
                            current_status_dictionary[key] = value # Keep as string if not JSON parsable

                except Exception as e:
                    logging.error(f"[{session_id}] - Error applying status update for key '{key}': {e}")
        else:
            # Add other UI commands to the list to be sent to the client
            final_actions_for_client.append(action)
    
    logging.info(f"[{session_id}] - Status Dictionary after updates: {current_status_dictionary}")
    
    # Return the remaining UI actions to the client, along with the *current* status for frontend rendering
    return jsonify({"actions": final_actions_for_client, "current_status": current_status_dictionary})

if __name__ == '__main__':
    app.run(debug=True)