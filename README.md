# Twitter Sentiment Analysis

This project performs **sentiment analysis** on Twitter data using machine learning techniques. The goal is to classify tweets as **positive** or **negative** based on their textual content.  

The project uses **Python**, **scikit-learn**, **NLTK**, and **TF-IDF** for feature extraction.

---

## Dataset

The dataset used is the **Sentiment140 dataset** from Kaggle:

- **Link:** [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)  
- **Size:** 1,600,000 tweets  
- **Columns:** `target`, `id`, `date`, `flag`, `user`, `text`  
- **Target:**  
  - `0` → Negative tweet  
  - `4` → Positive tweet (converted to `1` for processing)

The dataset is **balanced**, with 800,000 positive and 800,000 negative tweets.

---

## Project Steps

### 1. Data Preprocessing

- Removed punctuation, numbers, and special characters  
- Converted text to lowercase  
- Removed **stopwords** using NLTK  
- Applied **stemming** (PorterStemmer) to reduce words to their root form  

Example:  
```

Original: "I am loving the new update!"
Processed: "love new updat"

````

### 2. Feature Extraction

- Used **TF-IDF Vectorizer** to convert text into numerical vectors  
- Training data: `fit_transform()`  
- Test data: `transform()` (to prevent data leakage)

### 3. Train-Test Split

- Split data into **training (80%)** and **test (20%)** sets  
- Ensured **stratification** to maintain class balance  

### 4. Machine Learning Model

- Used **Logistic Regression** with `max_iter=1000`  
- Trained on the processed training data  
- Evaluated using **accuracy score**  

**Results:**  
- Training Accuracy: **79.87%**  
- Test Accuracy: **77.67%**

---

### 5. Save & Load Model

- Model saved using **pickle**: `trained_model.sav`  
- Can be loaded later for **predicting new tweets**  

Example prediction:
```python
prediction = loaded_model.predict(new_tweet_vector)
if prediction[0] == 0:
    print("Negative feed")
else:
    print("Positive feed")
````

---

## Technologies Used

* Python 3.x
* pandas, numpy
* scikit-learn (LogisticRegression, TfidfVectorizer)
* NLTK (stopwords, PorterStemmer)
* pickle (for saving and loading models)

---

## How to Run

1. Clone the repository
2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Place `kaggle.json` in `~/.kaggle/` and download the dataset:

```bash
kaggle datasets download -d kazanova/sentiment140
```

4. Extract the dataset:

```bash
unzip sentiment140.zip
```

5. Run `twitter_sentiment_analysis.ipynb` or your Python script

---

## References

1. [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
2. [NLTK Documentation](https://www.nltk.org/)
3. [scikit-learn Documentation](https://scikit-learn.org/)

```
