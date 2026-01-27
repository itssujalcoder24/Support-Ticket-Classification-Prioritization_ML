# Support Ticket Classification & Prioritization System
**Author:** Sujal Das (ssd)  
**Role:** Engineering Student | AI / ML Enthusiast  
<br>
---
 
## Project Overview
Modern organizations receive a large number of customer and internal support tickets every day.  
Manually reading, categorizing, and prioritizing these tickets is time-consuming, inconsistent, and slows down response times.
This project implements an **NLP-based Machine Learning system** that automatically:
- Reads raw text-based support tickets  
- Classifies them into predefined categories  
- Assigns a priority level (**High / Medium / Low**)  
- Helps support teams respond faster and manage workload efficiently  
The system is designed as a **decision-support tool** that assists human support agents rather than replacing them.
<br>
---

## Objectives
- Work with real-world text-based support ticket data  
- Clean and preprocess raw textual data  
- Convert text into numerical features using NLP techniques  
- Train and evaluate machine learning classification models  
- Design a priority assignment mechanism based on business logic  
- Demonstrate how ML can optimize support operations  
<br>
---

## Tech Stack
### Programming Language
- **Python 3.x**
### Libraries & Tools
- **pandas** – Data handling and preprocessing  
- **scikit-learn** – Machine learning models and evaluation  
- **matplotlib** – Visualization (confusion matrix)  
- **joblib** – Model persistence (saving/loading models)  
- **re** – Text preprocessing using regular expressions  
### ML & NLP Techniques
- Text cleaning and normalization  
- Token-based feature extraction  
- **TF-IDF Vectorization** (unigrams + bigrams)  
- **Multinomial Naive Bayes** classifier  
- Train/Test split with stratification  
- Model evaluation using accuracy, precision, recall, and F1-score  
<br>
---

## How to Run the Project
### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
<br>

### 2. Train the Model
```bash
python src/train_model.py
```

This step:
Preprocesses the dataset
Trains the TF-IDF + Naive Bayes classifier
Saves the trained model and vectorizer to the models/ directory
<br>

### 3️. Evaluate the Model
```bash
python src/evaluate.py
```

This generates:
Classification report (precision, recall, F1-score)
Confusion matrix for class-wise performance analysis
<br>

### 4. Run the Live Demo
```bash
python demo/demo_app.py
```

This launches an interactive command-line demo where users can enter a support ticket and receive predictions instantly.
<br>

### Model Performance Summary
Overall Accuracy: ~78%
Strong performance on:
Access
Hardware
Purchase
Storage
Expected confusion between semantically similar categories
(e.g., Access vs Administrative Rights)
<br>
This performance represents a realistic baseline model trained on real-world enterprise support ticket data.
<br>

### Priority Assignment Logic
The dataset does not contain explicit priority labels.
To handle this, priority is assigned using rule-based business logic, similar to how real support systems operate.<br>
This hybrid ML + rule-based approach ensures:
Explainability
Reliability
Immediate business usability
<br>

### Business Impact
This system helps organizations to:
Reduce manual ticket triage effort
Automatically route tickets to the correct teams
Identify high-urgency issues instantly
Improve SLA compliance and response times
Reduce backlog and support agent workload
The model functions as a decision-support system, enabling faster and smarter support operations.
<br>

### Future Enhancements
Train a dedicated ML model for priority prediction
Introduce class-weighted learning to handle imbalance
Experiment with advanced classifiers (Logistic Regression, SVM)
Deploy as a REST API using Flask or FastAPI
Build a web-based dashboard for visualization