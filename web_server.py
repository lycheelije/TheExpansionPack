from flask import Flask, request, jsonify

app = Flask(__name__)

# this is a placeholder and will be replaced with the function from the main library
def predict_response(message, prompt):
    response = f"Received message: '{message}' and prompt: '{prompt}', and generated a response."
    return response

@app.route('/your_endpoint', methods=['POST'])
def process_request():
    try:
        data = request.get_json()
        message = data.get('message', '')
        prompt = data.get('prompt', '')

        # Call the predict_response function with 'message' and 'prompt' as input
        response = predict_response(message, prompt)

        # Return the response as JSON
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    # Serve the index.html file from a directory named 'static'
    return send_from_directory('static', 'index.html')
    
if __name__ == '__main__':
    app.run(host='localhost', port=your_port)