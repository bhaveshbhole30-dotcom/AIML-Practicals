# AIML Practical Assignments – 10 Practicals

This repository contains runnable implementations of the 10 AIML practical assignments.

## Practicals

1. Identify the association between dependent and independent variables.
2. Apply clustering techniques using unsupervised learning.
3. Apply dimensionality reduction using PCA.
4. Apply Naive Bayes classification algorithms.
5. Apply the KNN algorithm for classification.
6. Implement Linear Regression.
7. Implement SVM for Regression.
8. Apply regularization to avoid overfitting.
9. Apply object detection from CCTV footage.
10. Design a model to detect Fake News from Multimodal data.

## Recommended setup in VS Code

Use Python 3.11 or newer. Create a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run any practical from its folder, for example:

```bash
cd practical_01_association
python practical_01.py
```

Each practical has:
- `practical_XX.py` – main executable program
- `README.md` – aim, theory, procedure, and result
- `data/` – small self-contained dataset where applicable
- `outputs/` – generated plots/results (created when the program runs)

## Important note for Practicals 9 and 10

Practical 9 uses YOLO object detection. Place a CCTV/video file in:
`practical_09_object_detection_cctv/data/input.mp4`

If no video is supplied, the script prints the exact setup instructions.

Practical 10 demonstrates a lightweight multimodal fake-news classifier using:
- text TF-IDF features
- image-derived visual features

It is an educational practical implementation, not a production fact-checking system.

## Suggested GitHub workflow

After verifying the programs locally:

```bash
git init
git add .
git commit -m "Add all 10 AIML practicals"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/aiml-practicals.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username and create the empty repository first.

## Academic submission

Before submission, run the practicals and keep the generated figures/results. Add your name, PRN, roll number, division, and academic year wherever your college format requires them.
