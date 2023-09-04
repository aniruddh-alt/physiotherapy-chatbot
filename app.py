from flask import Flask, render_template,jsonify
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flaskext.mysql import MySQL
import openai
from flask import Flask, request, jsonify


openai.api_key=""

app = Flask(__name__)
inputs=[]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chat')
def index():
    return render_template('chatbot.html')
sessions={}
@app.route('/ask', methods=['POST'])
def ask_question():
    user_input = request.form['user_input']
    conversation = []
    session_id = request.form.get('session_id')

    if session_id not in sessions:
        sessions[session_id] = {'conversation_history': []}

    bot_response = "I'm a chatbot. How can I assist you today?"

    sessions[session_id]['conversation_history'].append({'role': 'bot', 'message': bot_response})

    conversation.append({"role": "user", "content": user_input})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  
        messages=conversation,
    )

    bot_response = response.choices[0].message["content"]
    return jsonify({"bot_response": bot_response})



if __name__ == '__main__':
    app.run(debug=True)