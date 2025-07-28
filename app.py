from flask import Flask, request, jsonify, render_template, Response
from tutor import ask_tutor
import json # Import the json library

app = Flask(__name__)

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/api/courses')
def get_courses():
    """Serves the course data from the JSON file."""
    with open('courses.json', 'r') as f:
        courses_data = json.load(f)
    return jsonify(courses_data)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    context = data.get('context', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    def generate():
        for chunk in ask_tutor(question, context):
            yield chunk

    return Response(generate(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)