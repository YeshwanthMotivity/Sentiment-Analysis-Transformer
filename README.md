## 📊 Sentiment-Analysis-Transformer
Sentiment-Analysis-Transformer is a machine learning project designed to perform sentiment analysis on text data. It utilizes a Transformer-based model from the Hugging Face library to accurately classify text into positive, negative, or neutral sentiments. The project is optimized for efficient handling of large datasets in Parquet format.

---

## 🚀 Features
1. **Hugging Face Transformers:** Leverages the power of pre-trained models like BERT and DistilBERT for high-accuracy sentiment classification.
2. **Efficient Data Handling:** Uses the Hugging Face Datasets library and Pandas for seamless management of large Parquet files.
3. **Customizable:** The AutoModelForSequenceClassification makes it easy to fine-tune the model for any specific text classification task.
4. **Lightweight:** The project is optimized for local execution, making it accessible for quick experimentation and development.
5. **Scalable:** The underlying framework can be adapted to handle different datasets and a variety of text classification use cases.

---
### 🛠️ Tech Stack
| Component          |       Tool / Library         |                        Purpose                                 |
| ------------------ | ---------------------------- | -------------------------------------------------------------- |
| **Embeddings**     | `SentenceTransformer`        | Creates semantic vectors from text.                            |
| **Vector Search**  | `FAISS`                      | Provides fast and efficient vector similarity search.          |
| **Language Model** | `DialoGPT-small`             | Generates conversational responses based on retrieved context. |
| **Development**    | `Python`, `Jupyter Notebook` | The core programming language and development environment.     |

---

### 📄 How It Works
1. **Data Loading**: A Parquet file containing text and sentiment labels is loaded into the project.
2. **Dataset Conversion**: The data is converted into a Hugging Face Dataset object, which is optimized for transformer training.
3. **Tokenization**: An AutoTokenizer is used to convert the raw text into a format the model can understand.
4. **Fine-tuning**: A pre-trained transformer model is fine-tuned for sentiment analysis using AutoModelForSequenceClassification.
5. **Training**: The model is trained using the efficient Hugging Face Trainer API.
6. **Prediction**: The trained model can then be used to predict the sentiment of new, unseen text inputs.

---
## 🗂 Requirements
**Key Dependencies**
![req](https://github.com/user-attachments/assets/ad42abe0-223c-409d-9aca-22fc69947c8e)

## 💻 How to Run
1. **Clone the repository**:
git clone https://github.com/YeshwanthMotivity/Sentiment-Analysis-Transformer.git
cd Sentiment-Analysis-Transformer

2. **Launch the notebook**:
Open Sentiment.ipynb in Jupyter Notebook.
Execute the cells step by step for training and predictions.

---
## 📸 Architecture Diagram
![img6](https://github.com/user-attachments/assets/c3c63fb7-b0ce-4ad0-9894-3274488586df)

---
## 📈 Future Scope
1. **Real-time Deployment**: Deploy the model using a framework like Streamlit, Flask, or FastAPI to perform sentiment analysis in real time.
2. **Multi-label Classification**: Extend the project to handle more complex sentiment categories (e.g., joy, anger, surprise).
3. **Live Data Input**: Integrate the model with web forms or chatbots to analyze sentiment from live user inputs.

---
## 📬 Contact
For questions, reach out at:

Email 📧: yeshwanth.mudimala@motivitylabs.com

