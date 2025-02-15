# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Model Name**: Income Prediction Model
- **Model Version**: 1.0
- **Model Type**: RandomForestClassifier
- **Author**: [Your Name]
- **Date**: [YYYY-MM-DD]

## Intended Use
- **Primary Use**: Predict whether an individual's income exceeds $50,000/year based on census data.
- **Target Audience**: Data scientists, policymakers, and researchers.
- **Out-of-Scope Use Cases**: This model is not intended for use in high-stakes decision-making (e.g., loan approvals, hiring decisions).

## Training Data
- **Dataset**: UCI Census Income Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/census+income)
- **Preprocessing**: Data was preprocessed using one-hot encoding for categorical features and label binarization for the target variable (`salary`).
- **Train-Test Split**: 80% training, 20% testing.

## Evaluation Data
- **Dataset**: Test split from the UCI Census Income Dataset.
- **Metrics**:
  - Precision: 0.7391
  - Recall: 0.6384
  - F1 Score: 0.6851

## Metrics
- **Performance on Test Data**:
  - Precision: 0.7391
  - Recall: 0.6384
  - F1 Score: 0.6851
- **Performance on Data Slices**:
  The model's performance was evaluated on slices of categorical features. Below are some key metrics:
  
  #### **Workclass**
  - `Federal-gov`: Precision: 0.7971 | Recall: 0.7857 | F1: 0.7914
  - `Private`: Precision: 0.7362 | Recall: 0.6384 | F1: 0.6838
  - `Self-emp-inc`: Precision: 0.7586 | Recall: 0.7458 | F1: 0.7521

  #### **Education**
  - `Bachelors`: Precision: 0.7569 | Recall: 0.7333 | F1: 0.7449
  - `Masters`: Precision: 0.8263 | Recall: 0.8502 | F1: 0.8381
  - `HS-grad`: Precision: 0.6460 | Recall: 0.4232 | F1: 0.5114

  #### **Marital Status**
  - `Married-civ-spouse`: Precision: 0.7317 | Recall: 0.6922 | F1: 0.7114
  - `Never-married`: Precision: 0.8148 | Recall: 0.4272 | F1: 0.5605

  #### **Race**
  - `White`: Precision: 0.7372 | Recall: 0.6366 | F1: 0.6832
  - `Black`: Precision: 0.7407 | Recall: 0.6154 | F1: 0.6723

  #### **Sex**
  - `Male`: Precision: 0.7410 | Recall: 0.6607 | F1: 0.6985
  - `Female`: Precision: 0.7256 | Recall: 0.5107 | F1: 0.5995

  For a full list of performance metrics on all slices, see `slice_output.txt`.

## Ethical Considerations
- **Bias**: The model may reflect biases present in the training data (e.g., gender, race, or socioeconomic status). For example:
  - The model performs better for males (F1: 0.6985) than females (F1: 0.5995).
  - Certain racial groups (e.g., `White`) have higher F1 scores compared to others (e.g., `Black`).
- **Fairness**: Care should be taken to ensure the model does not disproportionately affect marginalized groups. For instance:
  - The model has lower recall for females (0.5107) compared to males (0.6607), indicating potential gender bias.
- **Transparency**: The model's predictions should be explainable to stakeholders. Efforts should be made to provide interpretable explanations for predictions.

## Caveats and Recommendations
- **Limitations**:
  - The model's performance may degrade on data that differs significantly from the training distribution.
  - The dataset is imbalanced, with fewer high-income individuals, which may affect the model's ability to generalize.
- **Recommendations**:
  - Regularly monitor the model's performance and retrain it with updated data to ensure accuracy.
  - Conduct fairness audits to identify and mitigate biases in the model.
  - Use the model in conjunction with human judgment for high-stakes decisions.
