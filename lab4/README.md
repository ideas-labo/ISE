# Lab 4: AI Model Fairness Testing
## Dataset details

This table provides detailed descriptions of various datasets, focusing on their sensitive attributes and target labels.

| **Dataset**           | **Sensitive Attributes**   | **Target Label**          | **Class**   | 
|-----------------------|----------------------------|---------------------------|-------------|
| **ADULT**             | gender, race, age          | Class-label               | Binary      |
| **COMPAS**            | Sex, Race                  | Recidivism                | Binary      | 
| **LAW SCHOOL**        | male,race                  | pass_bar                  | Binary      | 
| **KDD**               | sex, race                  | income                    | Binary      | 
| **DUTCH**             | sex,age                    | occupation                | Binary      | 
| **CREDIT**            | SEX,EDUCATION,MARRIAGE     | class                     | Binary      | 
| **CRIME**             | Black,FemalePctDiv         | class                     | Multi       | 
| **GERMAN**            | PersonStatusSex,AgeInYears | CREDITRATING              | Binary      | 

---

This table summarizes the sensitive attributes and target labels for each dataset, which are important for fairness evaluations in DNN models.


## Trained neural network details
Each dataset has a correspondingly named serialized DNN model that can be loaded and used for inference or further analysis without retraining.

**Change the DNN model**:
     - Replace the DNN model with the appropriate **serialized model** file.
     - Ensure that the model is loaded using the correct method (e.g., `keras.models.load_model` for Keras models).
    
     ```python
     from keras.models import load_model
     model = load_model('DNN/model_processed_adult.h5')
     ```
