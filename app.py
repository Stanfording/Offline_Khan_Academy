from flask import Flask, request, jsonify, render_template, Response
from tutor import ask_tutor
import json
import uuid # For generating simple session IDs

app = Flask(__name__)

# Dictionary to store conversation contexts per session
# In a real application, consider Flask sessions or a more persistent store (database, Redis)
user_contexts = {}

@app.route('/')
def home():
    """Serves the home HTML page."""
    return render_template('home.html')

@app.route('/classroom')
def classroom():
    """Serves the classroom HTML page."""
    # A new session ID is generated for each new classroom page load if not provided
    session_id = request.args.get('session_id', str(uuid.uuid4()))
    return render_template('classroom.html', session_id=session_id)

@app.route('/api/courses')
def get_courses():
    """Serves the course data from the JSON file."""
    try:
        with open('courses.json', 'r') as f:
            courses_data = json.load(f)
        return jsonify(courses_data)
    except FileNotFoundError:
        return jsonify({"error": "courses.json not found"}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "Error decoding courses.json"}), 500

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    session_id = data.get('session_id') # Get session ID from request

    if not question:
        # Allow empty question for initial prompt (Mr. Delight's greeting)
        # return jsonify({'error': 'No question provided'}), 400
        pass
    if not session_id:
        return jsonify({'error': 'No session_id provided'}), 400

    # Retrieve context for the current session, or start fresh if none exists
    current_context = user_contexts.get(session_id)

    def generate():
        last_chunk_context = None
        # The ask_tutor generator will yield content chunks, and finally a dict with "done" and "new_context"
        for chunk_data in ask_tutor(question, current_context):
            if isinstance(chunk_data, dict):
                if chunk_data.get("done"):
                    last_chunk_context = chunk_data.get("new_context")
                    break # All content yielded, context obtained, stop iteration
                elif "error" in chunk_data:
                    yield json.dumps({"error": chunk_data["error"]}) # Send error as JSON
                    break
            else:
                yield chunk_data # Yield text content
        
        # After the loop finishes (either due to break or generator exhaustion)
        # store the last received context
        if last_chunk_context is not None:
            user_contexts[session_id] = last_chunk_context
        # If no new context was provided (e.g., initial empty message before context is generated),
        # or if the model response was an error, the old context remains or stays None.
        # This is fine; the next request will use whatever is in user_contexts[session_id].

    return Response(generate(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)