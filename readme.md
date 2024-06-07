# Electricity Price Prediction

This notebook presents a comprehensive analysis of predicting the price of electricity in Spain using various machine learning algorithms. The main focus is on exploring different techniques, including Linear Regression, XGBRegressor, LSTM, Bidirectional LSTM, and Convolutional 1D models. PyTorch is employed for the deep learning experiment.

## Introduction
Electricity price prediction is a crucial task in the energy sector, as it enables stakeholders to make informed decisions about energy production, consumption, and trading. Accurate price forecasting helps utilities, grid operators, and consumers optimize resource allocation and manage costs effectively.

## Dataset

The dataset used in this project is available on google drive and can be downloaded using `gdown`, and consists of two main parts:

 - **Energy Generation Data**: This dataset contains historical information about energy generation, likely including details about different sources of energy generation, their capacities, and outputs over time.

 - **Weather Data**: The weather dataset includes historical weather information such as temperature, wind speed, humidity, and other relevant factors that may impact electricity prices.

Both datasets are crucial for predicting electricity prices as they provide valuable insights into the factors affecting energy production and consumption.

## Dependencies

The following dependencies are required to run the notebook:

- Python==3.11
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- torch==2.3.0+cu121

You can install the required libraries using pip:

```
pip install pandas numpy matplotlib scikit-learn
```

## Notebook Overview

The notebook covers the following main topics:

1. **Data Exploration and Preprocessing**: Exploring the dataset, handling missing values, and preprocessing the data for model training.

2. **Feature Engineering**: Creating additional features from the existing ones to improve model performance.

3. **Model Selection and Training**: Trying out different machine learning algorithms such as Linear Regression, Random Forest, and Gradient Boosting Regression for predicting electricity prices.

4. **Model Evaluation**: Evaluating the performance of each model using Root Mean Squared Error (RMSE).

5. **Hyperparameter Tuning**: Fine-tuning the hyperparameters of the best-performing model to optimize its performance further.

6. **Conclusion**: Summarizing the findings and suggesting potential areas for improvement or further research.

## How to Use

To use the notebook, follow these steps:

1. Clone or download this repository to your local machine.
2. Install the required dependencies as mentioned above.
3. Open the Jupyter Notebook using the following command:
   ```
   jupyter notebook "Electricity Price Pred.ipynb"
   ```
4. Execute the cells in the notebook sequentially to reproduce the analysis and results.

## License

The dataset and code in this repository are provided under the MIT License. Feel free to use and modify them for your own projects. If you find this work helpful, consider giving it a star on GitHub!


credit: [electricity price forecasting](https://www.kaggle.com/code/dimitriosroussis/electricity-price-forecasting-with-dnns-eda)
