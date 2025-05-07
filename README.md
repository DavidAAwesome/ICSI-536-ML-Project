# Bone Fracture Image Classifier

```
/ICSI-536-ML-PROJECT/
├── data/
├── nottensorflow/
├── extract_data.py
├── main.py
└── requirements.txt
```

## Install packages

There is a list of packages this code uses in `requirements.txt`. Activate your Python virtual environment, then run

```sh
pip install -r requirements.txt
```

to ensure that all dependencies are satisfied.

## Dataset

Our dataset is accessed from the `data/` directory, but it is _currently empty_. The original dataset we used can be found [here (original)](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project/code). A version of it already extracted into array form can be found [here (extracted)](https://kaggle.com/datasets/9c5df99a18584ce4ff676edb7a3ca0e22a86f535df2b5e53a282f71f750a089e).

To run the image processing script, put `BoneFractureYolo8/` under `data/`, then run `extract_data.py`.

To run main script, put `v8_test_extracted.npz`, and `v8_train_extracted.npz` under `data/`.

```
data/
├── BoneFractureYolo8/
├── v8_test_extracted.npz
└── v8_train_extracted.npz
```

## Running the main script

The training and evaluation of our model is done in `main.py`.

From the project root, run:

```sh
python main.py
```

Or better, if using Visual Studio Code with Python Extension, just click "Run Python File" when opening `main.py`.
