# Rice, Pistachio, and Grapevine Leaf Classification

This project implements a machine learning pipeline to classify images of rice, pistachio, and grapevine leaves. The goal is to accurately identify the plant species based on leaf images using computer vision techniques.

## Features

- Image preprocessing and augmentation
- Feature extraction using deep learning models
- Model training and evaluation
- Performance metrics and visualization

## Dataset

The dataset consists of labeled images of rice, pistachio, and grapevine leaves. Ensure the dataset is organized in separate folders for each class.

## Model Storage

All trained models are saved in the `Models/` directory. You can find your saved model files there after training.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/SARATH062005/Rice-Variety-Prediction.git
    cd rice-pistachio-and-grapevine-leaf-classification
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your dataset in the `data/` directory.
2. Run the training script:
    ```bash
    python train.py
    ```
3. Evaluate the model:
    ```bash
    python predict.py
    ```

## Results

- Accuracy, precision, recall, and confusion matrix are reported after training.
- Example predictions and visualizations are available in the `runs/` folder.

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License.