import re
from flask import Flask, request, render_template_string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Base de conhecimento
faq = {
    "como instalo um programa": "Para instalar um programa, você pode utilizar o gerenciador de pacotes do seu sistema. Por exemplo: sudo apt install nome-do-programa",
    "como remover um programa": "Você pode remover programas com: sudo apt remove nome-do-programa",
    "como atualizo o sistema": "Use: sudo apt update && sudo apt upgrade",
    "como vejo programas instalados": "Você pode usar: dpkg --list ou flatpak list",
    "o que você pode fazer": "Posso responder perguntas técnicas relacionadas a gerenciamento de pacotes e operações básicas no sistema.",
}

# Preprocessamento
questions = list(faq.keys())
answers = list(faq.values())
vectorizer = CountVectorizer().fit(questions)
X = vectorizer.transform(questions)


def clean(text):
    return re.sub(r"[^a-zA-Zà-ü0-9\s]", "", text.lower())


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChatBot - Ajuda Técnica</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f7fa;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            color: #333;
        }

        .chat-container {
            background-color: #fff;
            width: 100%;
            max-width: 600px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin: 20px;
        }

        h2 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 20px;
        }

        .chat-form {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }

        input[type="text"] {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }

        input[type="text"]:focus {
            border-color: #3498db;
        }

        button {
            padding: 10px 20px;
            background-color: #3498db;
            color: #fff;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        button:hover {
            background-color: #2980b9;
        }

        .chat-messages {
            max-height: 300px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #eee;
            border-radius: 5px;
            background-color: #fafafa;
            margin-bottom: 20px;
        }

        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            max-width: 80%;
        }

        .user-message {
            background-color: #e1f5fe;
            margin-left: auto;
            text-align: right;
        }

        .bot-message {
            background-color: #f1f1f1;
            margin-right: auto;
        }

        @media (max-width: 500px) {
            .chat-container {
                margin: 10px;
                padding: 15px;
            }

            .chat-form {
                flex-direction: column;
            }

            button {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <h2>ChatBot - Ajuda Técnica</h2>
        <div class="chat-messages">
            {% if user_input %}
                <div class="message user-message">
                    <strong>Você:</strong> {{ user_input }}
                </div>
                <div class="message bot-message">
                    <strong>ChatBot:</strong> {{ response }}
                </div>
            {% endif %}
        </div>
        <form class="chat-form" method="POST">
            <input type="text" name="user_input" placeholder="Digite sua pergunta aqui..." required autofocus>
            <button type="submit">Enviar</button>
        </form>
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def chatbot():
    user_input = ""
    response = ""
    if request.method == "POST":
        user_input = request.form["user_input"]
        if clean(user_input) in ["sair", "quero sair"]:
            response = "Até mais! Qualquer coisa, estou por aqui."
        else:
            user_vec = vectorizer.transform([clean(user_input)])
            sims = cosine_similarity(user_vec, X)
            max_sim_idx = sims.argmax()
            if sims[0, max_sim_idx] > 0:
                response = answers[max_sim_idx]
            else:
                response = "Desculpe, não entendi sua pergunta. Poderia reformular?"

    return render_template_string(
        HTML_TEMPLATE, user_input=user_input, response=response
    )


if __name__ == "__main__":
    app.run(debug=True)
