import re
from flask import Flask, request, render_template_string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download("stopwords")
nltk.download("rslp")

stemmer = SnowballStemmer("portuguese")

app = Flask(__name__)


# Carrega perguntas e respostas do arquivo
def load_faq(filepath="faq.txt"):
    with open(filepath, encoding="utf-8") as f:
        lines = f.read().splitlines()

    questions, answers = [], []
    q, a = None, None
    for line in lines:
        if line.startswith("PERGUNTA:"):
            q = line.replace("PERGUNTA:", "").strip()
        elif line.startswith("RESPOSTA:"):
            a = line.replace("RESPOSTA:", "").strip()
        if q and a:
            questions.append(q)
            answers.append(a)
            q, a = None, None
    return questions, answers


# limpeza básica do texto
def clean(text):
    # Coloca tudo em minúsculo e remove pontuação
    text = re.sub(r"[^a-zA-Zà-ü0-9\s]", "", text.lower())

    # Divide em palavras
    words = text.split()

    # Remove stopwords em português
    stop_words = set(stopwords.words("portuguese"))
    stop_words.discard("não")
    filtered_words = [word for word in words if word not in stop_words]

    # Aplica stemming
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    return " ".join(stemmed_words)


# Preprocessamento
questions, answers = load_faq("faq.txt")
cleaned_questions = [clean(q) for q in questions]
vectorizer = CountVectorizer().fit(cleaned_questions)
X = vectorizer.transform(cleaned_questions)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ChatBot - Suporte Técnico</title>
  <style>
    * {
      box-sizing: border-box;
    }

    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background-color: #1e1e2f;
      color: #f0f0f0;
      margin: 0;
      padding: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
    }

    .chat-container {
      background-color: #2a2a40;
      width: 100%;
      max-width: 700px;
      border-radius: 12px;
      box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
      padding: 24px;
      display: flex;
      flex-direction: column;
    }

    h2 {
      text-align: center;
      margin-bottom: 24px;
      color: #00c6ff;
      font-weight: 500;
    }

    .chat-messages {
      flex: 1;
      overflow-y: auto;
      padding: 16px;
      background-color: #1e1e2f;
      border-radius: 8px;
      margin-bottom: 20px;
      max-height: 300px;
      border: 1px solid #3a3a5c;
    }

    .message {
      margin: 10px 0;
      padding: 12px 16px;
      border-radius: 8px;
      max-width: 80%;
      display: inline-block;
      line-height: 1.4;
    }

    .user-message {
      background-color: #0059ff;
      color: #fff;
      margin-left: auto;
      text-align: right;
    }

    .bot-message {
      background-color: #3a3a5c;
      color: #f0f0f0;
      margin-right: auto;
    }

    .chat-form {
      display: flex;
      gap: 10px;
    }

    input[type="text"] {
      flex: 1;
      padding: 12px;
      border-radius: 6px;
      border: 1px solid #444;
      background-color: #121224;
      color: #fff;
      font-size: 16px;
    }

    input[type="text"]::placeholder {
      color: #aaa;
    }

    button {
      padding: 12px 20px;
      background: linear-gradient(to right, #00c6ff, #0072ff);
      color: #fff;
      border: none;
      border-radius: 6px;
      font-size: 16px;
      cursor: pointer;
      transition: background 0.3s;
    }

    button:hover {
      background: linear-gradient(to right, #00a2e2, #0059ff);
    }

    @media (max-width: 600px) {
      .chat-container {
        padding: 16px;
        margin: 10px;
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
    <h2>🤖 Suporte Técnico</h2>
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
      <input type="text" name="user_input" placeholder="Digite sua dúvida técnica aqui..." required autofocus>
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
        if clean(user_input) in ["sair", "quero sair", "tchau", "encerrar"]:
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
