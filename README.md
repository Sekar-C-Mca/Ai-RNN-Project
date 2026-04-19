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

### Deploy to Hugging Face Spaces (Recommended) ⭐

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
   - Name: `ai-rnn-sentiment` (or your preferred name)
   - License: Choose appropriate license
   - Space SDK: Select "Docker"
   - Space hardware: Select "CPU basic" or higher
3. Clone the Space repository locally
4. Copy the following files to your Space repo:
   - `app.py`
   - `requirements.txt`
   - `simple_rnn_imdb.h5`
   - `Dockerfile`
5. Push to the repository:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push
   ```
6. Hugging Face will automatically build and deploy your app

### Deploy to Streamlit Cloud

⚠️ **Note**: Streamlit Cloud has compatibility issues with TensorFlow and Python 3.14. Use Hugging Face Spaces for better TensorFlow support.

~~1. Push your repository to GitHub
2. Go to https://streamlit.io/cloud
3. Sign in with your GitHub account
4. Click "New app" and select your repository
5. Set the main file to `app.py`
6. Click "Deploy"~~

### Deploy with Docker Locally

Run locally with Docker:
```bash
docker build -t ai-rnn-sentiment .
docker run -p 8501:7860 ai-rnn-sentiment
```

Then visit `http://localhost:8501`

### Other Deployment Platforms

- **Railway**: Supports Docker deployments
- **Render**: Supports Docker and custom environments
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