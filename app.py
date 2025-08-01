from flask import Flask, request, jsonify, render_template
from tutor import ask_tutor
import json
import uuid
import logging
from dotenv import load_dotenv

# Load .env automatically
load_dotenv()

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

session_data = {}

INITIAL_STATUS_DICT = {
    "learning_confidence": 5,
    "learning_interest": 5,
    "learning_patience": 5,
    "effort_focus": 10,
    "weak_concept_spot": {},
    "current_lesson_progress": 0,
    "lesson_stage": "not_in_lesson",
    "current_lesson_title": "",
    "active_question_id": ""
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/classroom')
def classroom():
    session_id = request.args.get('session_id', str(uuid.uuid4()))
    if session_id not in session_data:
        session_data[session_id] = {"messages": [], "status": INITIAL_STATUS_DICT.copy()}
        logging.info(f"Initialized new session_id: {session_id}")
    return render_template('classroom.html', session_id=session_id)

@app.route('/api/courses')
def get_courses():
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

def clamp_status(status):
    for k in ["learning_confidence", "learning_interest", "learning_patience", "effort_focus"]:
        v = status.get(k)
        if isinstance(v, (int, float)):
            status[k] = max(1, min(10, v))
    if isinstance(status.get("current_lesson_progress"), (int, float)):
        status["current_lesson_progress"] = max(0, min(100, status["current_lesson_progress"]))

@app.route('/ask', methods=['POST'])
def ask():
    req_data = request.get_json()
    user_action = req_data.get('user_action')
    session_id = req_data.get('session_id')

    if not user_action or not isinstance(user_action, dict) or 'command' not in user_action:
        return jsonify({'error': 'Invalid or missing user_action'}), 400
    if not session_id:
        return jsonify({'error': 'No session_id provided'}), 400

    current_session = session_data.get(session_id)
    if current_session is None:
        current_session = {"messages": [], "status": INITIAL_STATUS_DICT.copy()}
        session_data[session_id] = current_session

    current_messages_history = current_session["messages"]
    current_status_dictionary = current_session["status"]

    llm_response_json, updated_messages_history = ask_tutor(
        user_action,
        current_status_dictionary,
        current_messages_history,
        session_id=session_id
    )

    if "error" in llm_response_json:
        return jsonify({"actions": [
            {"command": "ui_display_notes", "parameters": {"content": f"Oops! Mr. Delight seems a bit stuck. Error: {llm_response_json['error']}"}}
        ], "current_status": current_status_dictionary}), 500

    current_session["messages"] = updated_messages_history

    final_actions_for_client = []
    for action in llm_response_json.get("actions", []):
        if action.get("command") == "update_status":
            updates = action.get("parameters", {}).get("updates", {})
            for key, value in updates.items():
                try:
                    if isinstance(value, str) and (value.startswith('+') or value.startswith('-')):
                        delta = float(value)
                        if '.' in key:
                            parts = key.split('.')
                            temp = current_status_dictionary
                            for i, part in enumerate(parts):
                                if i == len(parts) - 1:
                                    if part not in temp:
                                        temp[part] = 0.0
                                    temp[part] = temp[part] + delta
                                else:
                                    temp = temp.setdefault(part, {})
                        else:
                            current_status_dictionary[key] = current_status_dictionary.get(key, 0.0) + delta
                    else:
                        try:
                            val = json.loads(value) if isinstance(value, str) and (value.startswith('{') or value.startswith('[')) else value
                            if '.' in key:
                                parts = key.split('.')
                                temp = current_status_dictionary
                                for i, part in enumerate(parts):
                                    if i == len(parts) - 1:
                                        temp[part] = val
                                    else:
                                        temp = temp.setdefault(part, {})
                            else:
                                current_status_dictionary[key] = val
                        except json.JSONDecodeError:
                            current_status_dictionary[key] = value
                except Exception as e:
                    logging.error(f"[{session_id}] - Error applying status update for key '{key}': {e}")
        else:
            final_actions_for_client.append(action)

    clamp_status(current_status_dictionary)
    stage = current_status_dictionary.get("lesson_stage")
    if isinstance(current_status_dictionary.get("current_lesson_progress"), (int, float)):
        if stage == "intuition" and current_status_dictionary["current_lesson_progress"] < 10:
            current_status_dictionary["current_lesson_progress"] = 10
        elif stage == "main_lesson" and current_status_dictionary["current_lesson_progress"] < 40:
            current_status_dictionary["current_lesson_progress"] = 40
        elif stage == "practice" and current_status_dictionary["current_lesson_progress"] < 70:
            current_status_dictionary["current_lesson_progress"] = 70
        elif stage == "lesson_complete":
            current_status_dictionary["current_lesson_progress"] = max(95, current_status_dictionary["current_lesson_progress"])

    return jsonify({"actions": final_actions_for_client, "current_status": current_status_dictionary})

if __name__ == '__main__':
    app.run(debug=True)