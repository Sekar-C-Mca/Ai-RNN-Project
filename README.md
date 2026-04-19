# IMDB Movie Review Sentiment Analysis with RNN

A deep learning project that uses a Recurrent Neural Network (RNN) to classify IMDB movie reviews as positive or negative. The project includes a Streamlit web application for easy sentiment analysis.

## Features

- **RNN Model**: Trained on IMDB dataset with Embedding layer and SimpleRNN
- **Streamlit Web App**: Interactive interface for sentiment prediction
- **Pre-trained Model**: Ready-to-use model for inference
- **Word Embedding**: Leverages word embeddings for better contextual understanding


## Installation

1. Clone the repository:
```bash
git clone https://github.com/Sekar-C-Mca/Ai-RNN-Project.git
cd Ai-RNN-Project
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Usage Instructions

1. Enter a movie review in the text area
2. Click the "Classify" button
3. View the sentiment (Positive/Negative) and prediction score

## Model Training

To train the model from scratch, run the `RNN.ipynb` notebook:

```bash
jupyter notebook RNN.ipynb
```

The notebook includes:
- Data loading and preprocessing
- Model architecture design
- Training with validation split
- Early stopping callback

## Deployment

### Deploy to Streamlit Cloud

1. Push your repository to GitHub
2. Go to https://streamlit.io/cloud
3. Sign in with your GitHub account
4. Click "New app" and select your repository
5. Set the main file to `app.py`
6. Click "Deploy"

### Deploy to Other Platforms

The project can be deployed to:
- **Heroku**: Requires Procfile and additional setup
- **AWS**: Using EC2 or Lambda
- **Google Cloud**: Using Cloud Run
- **Azure**: Using App Service

## Requirements

- Python 3.10+
- TensorFlow 2.13+
- Streamlit 1.28+
- NumPy, Pandas, scikit-learn

## Model Details

- **Architecture**: Embedding → SimpleRNN → Dense
- **Embedding Dimension**: 128
- **RNN Units**: 128 with ReLU activation
- **Input Sequence Length**: 500 (padded)
- **Vocabulary Size**: 10,000 words
- **Output**: Binary classification (Positive/Negative)

## Performance

- **Training Accuracy**: ~89%
- **Validation Accuracy**: ~86%
- **Test Accuracy**: ~85%

## License

This project is open source and available under the MIT License.

## Author

Sekar C - Mca Student

## Contact

Feel free to reach out for questions or improvements!