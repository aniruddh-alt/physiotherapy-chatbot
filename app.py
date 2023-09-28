from flask import Flask, render_template,jsonify
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flaskext.mysql import MySQL
import openai
import dotenv
from flask import Flask, request, jsonify
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.llms import OpenAI
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
from langchain.llms import OpenAI,OpenAIChat
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pandas as pd
from PIL import Image
import json
import openai
from dotenv import load_dotenv,find_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
import time
import PyPDF2
import os
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# # MySQL Configuration
# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'your_mysql_username'
# app.config['MYSQL_PASSWORD'] = 'your_mysql_password'
# app.config['MYSQL_DB'] = 'your_database_name'

# mysql = MySQL(app)


# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         hashed_password = generate_password_hash(password, method='sha256')
        
#         cur = mysql.connection.cursor()
#         cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
#         mysql.connection.commit()
#         cur.close()
        
#         flash('Registration successful. Please log in.', 'success')
#         return redirect(url_for('login'))
    
#     return render_template('register.html')

# @app.route('/register',methods=['POST','GET'])
# def login():
#     username=request.form['username']
#     password = request.form['password']
#     cur = mysql.connection.cursor()
#     cur.execute("SELECT * FROM users WHERE username = %s", [username])
#     user = cur.fetchone()
#     cur.close()
    
#     if user and check_password_hash(user['password'], password):
#         flash('Login successful', 'success')
#         return redirect(url_for('home'))
#     else:
#         flash('Invalid username or password', 'danger')
    
#     return render_template('login.html')















reader = PdfReader('ACL Excersises.pdf')

dotenv.load_dotenv()

llm = ChatOpenAI()
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)

from langchain.chains import ConversationChain
openai.api_key="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"

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
    # conversation = []
    # session_id = request.form.get('session_id')

    # if session_id not in sessions:
    #     sessions[session_id] = {'conversation_history': []}

    # bot_response = "I'm a chatbot. How can I assist you today?"

    # sessions[session_id]['conversation_history'].append({'role': 'bot', 'message': bot_response})

    # conversation.append({"role": "user", "content": user_input})

    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",  
    #     messages=conversation,
    # )

    # bot_response = response.choices[0].message["content"]
    # return jsonify({"bot_response": bot_response})
    prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    # Notice that we `return_messages=True` to fit into the MessagesPlaceholder
    # Notice that `"chat_history"` aligns with the MessagesPlaceholder name
    # memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    # conversation = LLMChain(
    #     llm=llm,
    #     prompt=prompt,
    #     verbose=True,
    #     memory=memory
    # )
    # # Notice that we just pass in the `question` variables - `chat_history` gets populated by memory
    # chain = ConversationChain(llm=llm,prompt=prompt)
    # retriever = vectorstore.as_retriever()
    # qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
    # print(chain(qa(user_input)))
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(ChatOpenAI(), chain_type="stuff")
    docs = docsearch.similarity_search(user_input)
    
    return jsonify({"bot_response": (chain.run(input_documents=docs, question=user_input))})


if __name__ == '__main__':
    app.run(debug=True)