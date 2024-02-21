import gradio as gr
import json
import os
import shutil
import magic
import ollama
# import subprocess
from unstructured.partition.pdf import partition_pdf
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores.qdrant import Qdrant
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

class GradioApp:
    def __init__(self):

        self.username = None
        self.password = None
        self.logged_in = False
        self.login_failed = False
        
        # self.username = "h3110Fr13nd"
        # self.password = "h3110Fr13nd"
        # self.logged_in = True
        # self.login_failed = False
        self.summarizer_prompt = """You are an Expert Doctor who analyses medical reports of patients, summarises it briefly using list and sections such as Overall Sumarry, Discharge Summary, Diaognosis, Treatment, Lab Reports and vitals wherever apply.
Use the following pieces of retrieved context to summarize the report. If you don't infer anything important, just say Everythin's Normal. 
Try to answer in 400 words maximum and keep the answer concise.
Give answer in Markdown Format.
"""
        self.rag_prompt_template = ChatPromptTemplate.from_template(
            """<s> [INST] You are a Medical Expert for question-answering tasks related to Medical Field. 
Use the following medical report to answer the question.
Try to answer in 400 words maximum. Use Markdown Format. 
[/INST] </s>
[INST] Question: {question} 
Context: {context} 
Answer: [/INST]
"""
        )
        self.ollama_client = ollama.Client(host="http://ollama:11434")
        self.embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
        self.vectorstore_url = "http://healthcare-rag-qdrant-store:6333"
        self.vectorstore = None
        self.collection_name = self.username
        self.llm = ChatOllama(model="mistral:latest", base_url="http://ollama:11434",)
        self.chain = None

        self.retriever = None
    def greet(self,name):
        return "Hello " + name + "!"
    
    def signup(self,username, password):
        with open("auth.json") as f:
            auth = json.load(f)
        if username in auth:
            return [
                # gr.Label("Logged in as ", visible=False),
                gr.Label("Signup failed. Username Exists Please Try Again. Or Login with another User.", visible=True),
                
                gr.Textbox(label="Username", value="", visible=True),
                gr.Textbox(label="Password", value="", visible=True),
                gr.Button("Login", visible=True),
                gr.Button("Signup", visible=True),
                gr.Button("Logout", visible=False)
            ]
        if len(username) < 4:
            return [
                # gr.Label("Logged in as ", visible=False),
                gr.Label("Signup failed. Username should be atleast 4 characters long.", visible=True),
                # gr.Label("Not logged in", visible=False),
                gr.Textbox(label="Username", value="", visible=True),
                gr.Textbox(label="Password", value="", visible=True),
                gr.Button("Login", visible=True),
                gr.Button("Signup", visible=True),
                gr.Button("Logout", visible=False)
            ]
        if len(password) < 8:
            return [
                # gr.Label("Logged in as ", visible=False),
                gr.Label("Signup failed. Password should be atleast 8 characters long.", visible=True),
                # gr.Label("Not logged in", visible=False),
                gr.Textbox(label="Username", value="", visible=True),
                gr.Textbox(label="Password", value="", visible=True),
                gr.Button("Login", visible=True),
                gr.Button("Signup", visible=True),
                gr.Button("Logout", visible=False)
            ]

        auth[username] = password
        with open("auth.json", "w") as f:
            json.dump(auth, f)
        self.username = username
        self.password = password
        # makedir for user data/username
        os.makedirs(f"data/{username}", exist_ok=True)
        return [
                # gr.Label("Logged in as ", visible=False),
                gr.Label("SignUp Successful. Please Log In.", visible=True),
                # gr.Label("Not logged in", visible=False),
                gr.Textbox(label="Username", value="", visible=True),
                gr.Textbox(label="Password", value="", visible=True),
                gr.Button("Login", visible=True),
                gr.Button("Signup", visible=True),
                gr.Button("Logout", visible=False)
            ]
    def login(self,username, password):
        with open("auth.json") as f:
            auth = json.load(f)
        if username in auth and auth[username] == password:
            self.username = username
            self.password = password
            self.logged_in = True
            self.login_failed = False
            return [
                gr.Label("Logged in as " + self.username, visible=True),
                # gr.Label("Login failed. Please Try Again", visible=False),
                # gr.Label("Not logged in", visible=False),
                gr.Textbox(label="Username", visible=False),
                gr.Textbox(label="Password", visible=False),
                gr.Button("Login", visible=False),
                gr.Button("Signup", visible=False),
                gr.Button("Logout", visible=True)
                ]
        else:
            return [
                # gr.Label("Logged in as ", visible=False),
                gr.Label("Login failed. Please Try Again. Or SignUp.", visible=True),
                # gr.Label("Not logged in", visible=False),
                gr.Textbox(label="Username", value="", visible=True),
                gr.Textbox(label="Password", value="", visible=True),
                gr.Button("Login", visible=True),
                gr.Button("Signup", visible=True),
                gr.Button("Logout", visible=False)
            ]
    def logout(self):
        self.username = None
        self.password = None
        self.logged_in = False
        return [
                # gr.Label("Logged in as ", visible=False),
                gr.Label("Not logged in", visible=True),
                gr.Textbox(label="Username", value="", visible=True),
                gr.Textbox(label="Password", value="", visible=True),
                gr.Button("Login", visible=True),
                gr.Button("Signup", visible=True),
                gr.Button("Logout", visible=False)
            ]
    
    def upload_files(self,files):
        file_paths = [file.name for file in files]
        current_dir = os.getcwd()
        new_file_paths = []
        output_text = "   "
        data_list = []
        for file in file_paths:
            file_path = current_dir + f"/data/{self.username}/" + os.path.basename(file)
            shutil.move(file, file_path)
            new_file_paths.append(file_path)
            mimetype = magic.from_file(file_path, mime=True)
            if mimetype == "application/pdf":
                # print("PDF detected")
                elements = partition_pdf(
                    filename=file_path,
                    infer_table_structure=True,
                    chunking_strategy="by_title",
                    max_characters=3000,
                    new_after_n_chars=2850,
                    combine_text_under_n_chars=1500,
                    overlap=True,
                    overlap_all=True,
                    ) 
                for elem in elements:
                    if "Table" in repr(elem):
                        output_text += "\n" + elem.to_dict()["metadata"]["text_as_html"] + "\n"
                        if len(data_list) > 0:
                            data_list[-1] += "\n\n" + elem.to_dict()["metadata"]["text_as_html"]+ "\n\n"
                        else:
                            data_list.append(elem.to_dict()["metadata"]["text_as_html"]+ "\n\n")
                    else:
                        if len(elem.text) < 100:
                            continue
                        output_text += "\n----------------------\n" + elem.text + "\n"
                        data_list.append(elem.text + "\n\n")

        data_summary = []
        for d in data_list:
            resp = self.ollama_client.generate(
                model="stablelm2:latest",
                prompt=self.summarizer_prompt + "Summarize the following medical report: " + d,
                system=self.summarizer_prompt,
                )
            data_summary.append(resp["response"])
        self.vectorstore = Qdrant.from_texts(
            data_summary,
            self.embeddings,
            url=self.vectorstore_url,
            prefer_grpc=False,
            collection_name=self.collection_name,
            force_recreate=True,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
        )
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.rag_prompt_template
                      | self.llm
                      | StrOutputParser())
        # with open(f"data/{self.username}/data.json", "w") as f:
        #     json.dump(data_list, f)
        # # self.processes.append(subprocess.Popen(["python", "infer.py", file_path]))
        return new_file_paths, gr.Markdown(label="Output", value=output_text)
    def user(self,user_message, history):
        return "", history + [[user_message, None]]
    
    def bot(self,history):
        print(history)
        prompt = history[-1][0] or ""
        # for chunks in self.chain.stream(prompt):
        #     history[-1][1] = history[-1][1] or ""
        #     history[-1][1] += chunks
        #     # yield "", history
        history[-1][1] = history[-1][1] or ""
        history[-1][1] += self.chain.invoke(prompt)
        print(history[-1][1])
        print(history)
        return history
# def init_app(self):
with gr.Blocks() as demo:
    gradio_app = GradioApp()
    # loggd_in_block = [
    #             gr.Label("Logged in as " + "h3110Fr13nd", visible=True),
    #             gr.Textbox(label="Username", visible=False),
    #             gr.Textbox(label="Password", visible=False),
    #             gr.Button("Login", visible=False),
    #             gr.Button("Signup", visible=False),
    #             gr.Button("Logout", visible=True)
                # ]
    login_status = gr.Label("Not logged in")
    username = gr.Textbox(label="Username")
    password = gr.Textbox( label="Password",type="password")
    login = gr.Button("Login")
    signup = gr.Button("Signup")
    logout = gr.Button("Logout", visible=False)
    login.click(gradio_app.login, [username, password], outputs=[login_status, username, password, login, signup, logout])
    signup.click(gradio_app.signup, [username, password], outputs=[login_status, username, password, login, signup, logout])
    logout.click(gradio_app.logout, outputs=[login_status, username, password, login, signup, logout])

    files = gr.Files(label="Upload Documents and Medical Reports", type="filepath", file_types=["pdf", "docx", "jpg", "jpeg", "png"])
    upload_button = gr.UploadButton(label="Upload Documents and Medical Reports", type="filepath", file_count='multiple', file_types=["pdf", "docx", "jpg", "jpeg", "png"], )
    output_text = gr.Markdown(label="Output", value="   ")
    infer_status = gr.Label("Infer Status: ", visible=False)


    upload_button.upload(gradio_app.upload_files, upload_button, [files, output_text])


    
        

    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    msg.submit(gradio_app.user, [msg, chatbot], [msg, chatbot], queue=False).then(
      gradio_app.bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

            
# demo.queue()
demo.launch(share=True)





# gradio_app.init_app()