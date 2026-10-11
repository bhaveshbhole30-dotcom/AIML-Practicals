# Practical 10: Fake News Detection from Multimodal Data

## Aim
Design a model that combines textual and visual information for fake-news classification.

## Theory
Multimodal learning combines information from more than one modality. Here:
- Text is converted into TF-IDF features.
- Each image is converted into simple visual features using a grayscale histogram and edge density.
- The two feature sets are concatenated and classified using Logistic Regression.

This is a compact, offline educational demonstration. The included data is a toy dataset created only so the practical runs immediately; it must not be presented as a real news dataset or as evidence about actual news stories.

## Procedure
1. Load the text/image/label CSV.
2. Extract TF-IDF text features.
3. Extract visual features from each image.
4. Concatenate both feature matrices.
5. Train a Logistic Regression classifier.
6. Evaluate accuracy and display a confusion matrix.

## Execution
```bash
python practical_10.py
```
