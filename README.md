# Sequence to Sequence Learning with LSTM

This project demonstrates a basic implementation of **Sequence-to-Sequence (Seq2Seq)** learning using **LSTM** networks in Keras. It translates **English sentences into Spanish** using a dataset of simple English-Spanish sentence pairs.

## 🧠 Objective

To understand how Seq2Seq models work using RNN-based architectures (specifically LSTMs) and apply them for basic machine translation tasks.

## 📁 Contents

- `Experiment_5_Sequence_To_Sequence_Learning_with_LSTM.ipynb`: Complete notebook with data loading, preprocessing, model training, and inference.
- English-Spanish sentence pairs (loaded from file during execution).

## 📌 Key Components

- **Data Preprocessing**:
  - Loads a CSV file with English-Spanish sentence pairs.
  - Tokenizes and pads sentences for encoder and decoder inputs.

- **Model Architecture**:
  - Encoder: Embedding + LSTM.
  - Decoder: Embedding + LSTM + Dense softmax.
  - Teacher forcing used during training.

- **Training**:
  - Categorical crossentropy loss.
  - Adam optimizer.
  - Trained with teacher forcing for multiple epochs.

- **Inference**:
  - Inference models built separately for encoder and decoder.
  - Predicts translations one word at a time using decoder states.

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/amank010/Deep-Learning-Seq2Seq.git
   cd Deep-Learning-Seq2Seq
   ```

2. Open the notebook:
   ```bash
   jupyter notebook Experiment_5_Sequence_To_Sequence_Learning_with_LSTM.ipynb
   ```

3. Run all cells sequentially to:
   - Load and preprocess data.
   - Train the model.
   - Translate new English sentences into Spanish.

## 🛠️ Dependencies

- Python 3.x
- NumPy
- TensorFlow / Keras
- Pandas
- Scikit-learn (for label encoding)
- Matplotlib (for plotting loss curves)

Install all dependencies using:

```bash
pip install numpy pandas scikit-learn matplotlib tensorflow
```

## 📌 Sample Result

Input: `go.`  
Output: `ve.`

Input: `i am hungry.`  
Output: `tengo hambre.`

## 📄 License

This project is for educational purposes.
