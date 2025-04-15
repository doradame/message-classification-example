# Message Classification Example

A Python app to play with some AI techniques classifying message replies.

This project demonstrates how to classify user replies to marketing messages (e.g., SMS or chat) using a series of natural language processing techniques:

🔹 Keyword-based classification  
🔹 Sentiment analysis  
🔹 Semantic similarity using embeddings  
🔹 Fallback classification with an LLM (e.g., OpenAI GPT)

It supports multilingual input (🇬🇧 English and 🇮🇹 Italian) and outputs:

- Detected intent (`interested`, `not_interested`, `unsubscribe`)
- Suggested reply
- Confidence score
- Detected language

---

## ⚙️ Installation

Clone the repository:

```bash
https://github.com/doradame/message-classification-example.git
cd message-classification-example
```

 Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

Install dependencies:
If you're running on CPU only, install torch with:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install the rest:

```bash
pip install -r requirements.txt
```

Configure your environment:

```bash
cp .env.example .env
nano .env  # Or use your preferred text editor
```

Inside `.env`, set your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

---

## 🚀 First Run

Make the script executable (just the first time) and then start the app:

```bash
chmod +x run.sh
./run.sh
```

Wait until it finish to download everything and setup, in something more than 1G so the first time it could take a lot of time

🌐 The API will be available at: `http://localhost:5001/classify`

---

##  Quick Test with curl

You can test the app using `curl`:

```bash
curl -X POST http://localhost:5001/classify \
-H "Content-Type: application/json" \
-d '{"message": "I don t want to hear anymore from Mojalab.com"}'
```

Expected JSON output:

```json
{"confidence":0.75,
 "intent":"unsubscribe",
 "language":"en",
 "reply":"Sorry to bother you. You won\u2019t hear from us again."}
```

---



---

Made with ❤️ by [MojaLab](https://mojalab.com) – Learn, test, build.
