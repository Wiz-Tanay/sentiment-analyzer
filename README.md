# 💬 Sentiment Analyzer (AI-Powered)

A real-time **Sentiment Analysis Web Application** that analyzes user input and predicts whether the sentiment is **Positive, Negative, or Neutral**, along with a confidence score.

🔗 **Live App:** https://tanay-sentiment-analyzer-321.streamlit.app

---

## 🚀 Features

* 🔍 Real-time sentiment prediction
* 😊 Supports emojis and hashtags
* 📊 Confidence score visualization
* 🎨 Modern, interactive UI (Streamlit)
* 🧠 Machine Learning-based predictions
* 🌐 Fully deployed web application

---

## 🧠 How It Works

1. User enters text input
2. Text is preprocessed:

   * Lowercasing
   * Emoji conversion
   * Hashtag cleaning
   * Noise removal
3. TF-IDF Vectorizer converts text into numerical features
4. Logistic Regression model predicts sentiment
5. Output is displayed with confidence score

---

## 🛠️ Tech Stack

* **Python**
* **Scikit-learn**
* **Pandas / NumPy**
* **Streamlit**
* **Emoji (for preprocessing)**

---

## 📂 Project Structure

```
sentiment-analyzer/
├── app.py              # Streamlit UI
├── train.py           # Model training script
├── model.pkl          # Trained ML model
├── tfidf.pkl          # TF-IDF vectorizer
├── requirements.txt   # Dependencies
└── Twitter_Data.csv   # Dataset (optional)
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/Wiz-Tanay/sentiment-analyzer.git
cd sentiment-analyzer
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the app

```
streamlit run app.py
```

---

## 📊 Model Details

* Algorithm: **Logistic Regression**
* Feature Extraction: **TF-IDF (with n-grams)**
* Dataset: **Twitter Sentiment Dataset**
* Accuracy: ~85%

---

## ⚠️ Limitations

* Does not fully understand sarcasm or deep context
* May misclassify nuanced or philosophical sentences
* Focuses on sentiment, not intent (e.g., violence detection)

---

## 🔮 Future Improvements

* Use transformer models (BERT) for better context understanding
* Improve slang handling
* Add multilingual support
* Enhance UI with more interactive elements

---

## 👨‍💻 Author

**Tanay Pandey**
GitHub: https://github.com/Wiz-Tanay

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
