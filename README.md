## Credit-Risk-Model---PD-to-EL

This is a Expected Credit Loss Model - It comprises a PD model, LGD model, EAD model, Credit Scoring and ultimately, ECL calculation

## 1. Introduction

This is a comprehensive overview of the Credit Risk Model designed to calculate Expected Credit Losses (ECL). The model leverages historical loan data to estimate key risk parameters: Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). These parameters are crucial for financial institutions to assess and manage credit risk in their portfolios, aligning with regulatory and/or accounting standards requirements such as IFRS 9.

## 2. Data Overview and Preprocessing

The initial steps involve loading this data and performing essential exploratory analysis and data cleaning.

### 2.1. Importing Libraries and Data

The process begins by importing necessary Python libraries, `numpy` for numerical operations and `pandas` for data manipulation. The primary dataset is then loaded into a pandas DataFrame.

### 2.2. Data Exploration

After loading, a quick inspection of the dataset's columns and initial rows helps in understanding its structure and content. The notebook displays all column names and the first/last few rows of the DataFrame.

### 2.3. Filtering for Defaulted Loans

For the purpose of calculating Loss Given Default (LGD) and Exposure at Default (EAD), the analysis focuses specifically on loans that have defaulted. This is achieved by filtering the `loan_status` column for 'Charged Off' and 'Does not meet the credit policy. Status:Charged Off' statuses.

**Code Snippet:**
```python
loan_data_defaults = loan_data_preprocessed[loan_data_preprocessed['loan_status'].isin(['Charged Off','Does not meet the credit policy. Status:Charged Off'])]
```

**Explanation:**
- The `.isin()` method is used to select rows where the `loan_status` column matches any of the specified default categories. This creates a new DataFrame, `loan_data_defaults`, containing only the defaulted loans.

### 2.4. Handling Missing Values for Independent Variables

Certain independent variables, specifically `mths_since_last_delinq` and `mths_since_last_record`, contain missing values. These are imputed with zeros.

## 3. Dependent Variables for LGD and EAD Models

This section focuses on creating the dependent variables required for the Loss Given Default (LGD) and Exposure at Default (EAD) models: `recovery_rate` and `CCF` (Credit Conversion Factor).

### 3.1. Calculating Recovery Rate

The recovery rate represents the proportion of the outstanding loan amount that a lender recovers after a default. It is calculated as the ratio of `recoveries` to `funded_amnt`.

**Code Snippet:**
```python
loan_data_defaults['recovery_rate'] = loan_data_defaults['recoveries'] / loan_data_defaults['funded_amnt']
loan_data_defaults['recovery_rate'].describe()
```

**Explanation:**
- A new column, `recovery_rate`, is created by dividing the `recoveries` amount by the `funded_amnt` for each defaulted loan. This directly quantifies the percentage of the loan that was recovered.

### 3.2. Adjusting Recovery Rate Values

Recovery rates are inherently bounded between 0 and 1 (or 0% and 100%). The code ensures that any calculated recovery rates falling outside this range are capped at these limits.

**Code Snippet:**
```python
loan_data_defaults['recovery_rate'] = np.where(loan_data_defaults['recovery_rate'] > 1, 1, loan_data_defaults['recovery_rate'])
loan_data_defaults['recovery_rate'] = np.where(loan_data_defaults['recovery_rate'] < 0, 0, loan_data_defaults['recovery_rate'])

```

**Explanation:**
- `np.where()` is a powerful function for conditional assignments. The first line sets any `recovery_rate` greater than 1 to 1, effectively capping it at 100%.
- The second line similarly sets any `recovery_rate` less than 0 to 0, ensuring no negative recovery rates. This step is crucial for maintaining the logical bounds of the recovery rate.

### 3.3. Calculating Credit Conversion Factor (CCF)

The Credit Conversion Factor (CCF) is a key component of the EAD model, representing the proportion of the undrawn loan amount that is likely to be drawn down before default. it is essential in estimatinig the likelihood and extent to which off-balance-sheet exposures will become actual on-balance sheet credit exposures. It's calculated based on the difference between the funded amount and the total principal received, relative to the funded amount.

**Code Snippet:**
```python
loan_data_defaults['CCF'] = (loan_data_defaults['funded_amnt'] - loan_data_defaults['total_rec_prncp']) / loan_data_defaults['funded_amnt']
```

**Explanation:**
- The `CCF` is derived by taking the difference between the `funded_amnt` (total loan amount committed) and `total_rec_prncp` (total principal received up to default), and then dividing this by the `funded_amnt`. This gives an indication of how much of the committed but undrawn amount was utilized before default.

### 3.4. Saving Processed Default Data

Finally, the DataFrame containing only defaulted loans with the newly calculated dependent variables (`recovery_rate` and `CCF`) is saved to a new CSV file for future use.


## 4. Exploring Dependent Variables Visually

Visualizing the distribution of the dependent variables is crucial for understanding their characteristics and informing subsequent modeling choices. This section uses histograms to achieve this.

### 4.1. Histogram of Recovery Rate

A histogram of the `recovery_rate` is plotted to visualize its distribution.

**Code Snippet:**
```python
plt.hist(loan_data_defaults['recovery_rate'], bins = 100)
plt.hist(loan_data_defaults['recovery_rate'], bins = 50)
```

**Explanation and Interpretation:**
- `plt.hist()` generates a histogram. The `bins` parameter controls the number of bins, which affects the granularity of the distribution visualization. Two histograms are generated, one with 100 bins and another with 50 bins, to observe the distribution at different levels of detail.

### 4.3. Histogram of Credit Conversion Factor (CCF)

A histogram of the `CCF` is plotted to understand its distribution.

**Code Snippet:**
```python
plt.hist(loan_data_defaults['CCF'], bins = 100)
```

**Explanation and Interpretation:**
- Similar to the recovery rate, `plt.hist()` is used to visualize the distribution of the `CCF` with 100 bins.

### 4.4. Creating a Binary Recovery Rate Variable

To facilitate a two-stage modeling approach for LGD (where the first stage predicts whether any recovery occurs), a binary variable `recovery_rate_0_1` is created. This variable is 0 if the recovery rate is 0, and 1 otherwise.

**Code Snippet:**
```python
loan_data_defaults['recovery_rate_0_1'] = np.where(loan_data_defaults['recovery_rate'] == 0, 0, 1)
```

**Explanation:**
This effectively transforms the continuous recovery rate into a categorical variable, which can be used as a target for our classification model, i.e. logistic regression, in the first stage of an LGD model. This approach helps to address the large number of zero recoveries often observed in credit datasets.



## LGD and EAD models

This section focuses on preparing the dependent variables for the LGD and EAD models.

## 5. Loss Given Default (LGD) Model

Loss Given Default (LGD) is the amount of money a bank loses when a borrower defaults, expressed as a percentage of the exposure at the time of default. Our LGD model is built in two stages: first, predicting whether any recovery occurs (a classification problem), and second, predicting the magnitude of recovery if it does occur (a regression problem).

### 5.1 Stage 1: Predicting Zero vs. Non-Zero Recovery

This stage uses logistic regression to predict the binary `recovery_rate_0_1` variable. This helps us identify loans where we expect to recover some amount versus those where we anticipate a complete loss.

#### 5.1.1 Data Preparation for Stage 1 LGD Model

We need to select the relevant features (independent variables) and the target variable for our logistic regression model.

```python
X = loan_data_defaults.drop(["recovery_rate","recovery_rate_0_1", "CCF"], axis = 1)
y = loan_data_defaults["recovery_rate_0_1"]
```

- **`X = ... .drop(...)`**: We create our feature set `X` by dropping the dependent variables (`recovery_rate`, `recovery_rate_0_1`, `CCF`) from the `loan_data_defaults` DataFrame. The `axis=1` indicates that we are dropping columns. The remaining columns in `X` are then used as predictors.
- **`y = ...`**: Our target variable `y` is set to `recovery_rate_0_1`, which is the binary indicator of whether any recovery occurred.

#### 5.1.2 Splitting Data into Training and Testing Sets

To properly evaluate our model, we split the data into training and testing sets. The model learns from the training data and is then evaluated on the unseen testing data.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```

**20% of the data is used for testing, and the remaining 80% for training our logistic regression model. `random_state = 42` ensures reproducibility of the split to safeguard getting the same split every time we run the code.

#### 5.1.3 Training the Logistic Regression Model

Now, we initialize and train our logistic regression model.

```python
log_reg_0_1 = LogisticRegression(solver=\'liblinear\', random_state=42)
log_reg_0_1.fit(X_train, y_train)
```

- **`LogisticRegression(...)`**: We create an instance of the logistic regression model. `solver=\'liblinear\'` specifies the algorithm used for optimization, which is suitable for small datasets and L1/L2 regularization. `random_state=42` ensures reproducibility of the model's internal randomness.
- **`.fit(X_train, y_train)`**: This is where the model learns the relationships between the features (`X_train`) and the target variable (`y_train`).

#### 5.1.4 Evaluating Stage 1 LGD Model Performance

After training, we evaluate how well our model predicts zero vs. non-zero recovery. We use `roc_auc_score` and `classification_report` for these.

```python
y_pred_proba_0_1 = log_reg_0_1.predict_proba(X_test)[:, 1]
print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba_0_1):.4f}")

y_pred_0_1 = log_reg_0_1.predict(X_test)
print(classification_report(y_test, y_pred_0_1))
```

- **`predict_proba(X_test)[:, 1]`**: This predicts the probabilities of the positive class (non-zero recovery, which is `1`) for the test set. We take `[:, 1]` to get only the probabilities for the positive class.
- **`roc_auc_score(y_test, y_pred_proba_0_1)`**: The Area Under the Receiver Operating Characteristic (ROC) Curve (AUC) is a common metric for binary classification models. An AUC of 0.5 suggests no discrimination (like random guessing), while 1.0 indicates perfect discrimination. A higher AUC is better. The AUC for our model = 0.6480801073622668
- **`predict(X_test)`**: This predicts the class labels (0 or 1) for the test set based on a default threshold (usually 0.5).
- **`classification_report(y_test, y_pred_0_1)`**: This provides a comprehensive summary of the model's performance, including precision, recall, f1-score, and support for each class (0 and 1). These metrics help us understand the trade-offs between correctly identifying positive cases and avoiding false positives.

### 5.2 Stage 2: Predicting Recovery Rate for Non-Zero Recoveries

For loans where we predicted a non-zero recovery (i.e., `recovery_rate_0_1` is 1), we built a linear regression model to predict the actual `recovery_rate`.

#### 5.2.1 Data Preparation for Stage 2 LGD Model

We filter the `loan_data_defaults` to include only those with non-zero recovery and then prepare the features and target variable.

```python
loan_data_defaults_nonzero_recovery = loan_data_defaults[loan_data_defaults["recovery_rate_0_1"] == 1]

X_lgd = loan_data_defaults_nonzero_recovery.drop(["recovery_rate","recovery_rate_0_1", "CCF"], axis = 1)
y_lgd = loan_data_defaults_nonzero_recovery["recovery_rate"]
```

- **`loan_data_defaults_nonzero_recovery = ...`**: This creates a new DataFrame containing only the defaulted loans where `recovery_rate_0_1` is 1.
- **`X_lgd = ... .drop(...)`**: Similar to Stage 1, we drop the dependent variables to form our feature set `X_lgd`.
- **`y_lgd = ...`**: Our target variable `y_lgd` is now the actual `recovery_rate` for these non-zero recovery cases.

#### 5.2.2 Splitting Data for Stage 2 LGD Model

Again, we split the data into training and testing sets for proper evaluation of the linear regression model.

```python
X_lgd_train, X_lgd_test, y_lgd_train, y_lgd_test = train_test_split(X_lgd, y_lgd, test_size = 0.2, random_state = 42)
```

- This performs the same splitting logic as in Stage 1, but on the `loan_data_defaults_nonzero_recovery` dataset.

#### 5.2.3 Training the Linear Regression Model

We use `LinearRegression` from `sklearn.linear_model` to predict the continuous `recovery_rate`.

```python
from sklearn.linear_model import LinearRegression

lin_reg_lgd = LinearRegression()
lin_reg_lgd.fit(X_lgd_train, y_lgd_train)
```

- **`LinearRegression()`**: Initializes the linear regression model.
- **`.fit(X_lgd_train, y_lgd_train)`**: Trains the model on the features and target from the non-zero recovery training set.

### 5.3 Combining LGD Model Stages

To get a single LGD prediction for all defaulted loans, we combine the results from both stages.

This two-stage approach is common in LGD modeling to handle the large number of zero recoveries, which can be challenging for a single regression model.

## 6. Exposure at Default (EAD) Model

Exposure at Default (EAD) is the total value of the loan that is outstanding at the time of default. For revolving credit facilities, this often involves modeling the Credit Conversion Factor (CCF), which is the proportion of the undrawn commitment that is drawn down at default. Our EAD model will focus on predicting this CCF.

### 6.1 Data Preparation for EAD Model

We will use the `loan_data_defaults` dataset, as EAD is relevant for defaulted loans. We need to define our features (independent variables) and the target variable (`CCF`).

### 6.2 Splitting Data into Training and Testing Sets

To ensure our EAD model is robust and generalizes well to unseen data, we split the dataset into training and testing subsets.

### 6.3 Training the Linear Regression Model for EAD

Since CCF is a continuous variable, we employ a linear regression model to predict its value.

## 7. Expected Credit Loss (ECL) Calculation

This section brings together the three key components we've modeled—Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD)—to calculate the Expected Credit Loss (ECL) for each loan. The ECL represents the weighted average of credit losses, where the weights are the probabilities of default. It's a crucial metric for financial institutions to provision for potential losses and assess their overall credit risk.

### 7.1 Calculating Individual ECL

The fundamental formula for Expected Credit Loss for a single exposure is:

$ECL = PD \times LGD \times EAD$

Where:
*   **PD (Probability of Default):** The likelihood that a borrower will default on their financial obligations over a specific time horizon. We derived this from our logistic regression model.
*   **LGD (Loss Given Default):** The proportion of the exposure that is lost if a default occurs. This was estimated using our linear regression model on the recovery rates.
*   **EAD (Exposure at Default):** The total value of the exposure that is expected to be outstanding at the time of default. We modeled this based on the `funded_amount` and `mths_since_last_delinq`.

The notebook calculates the ECL for each loan by multiplying the predicted PD, LGD, and EAD values. This provides a granular view of the expected loss for every individual in the portfolio.

### 7.2 Aggregating ECL

After calculating the individual ECL for each loan, the notebook proceeds to aggregate these values to understand the total expected credit loss across the entire portfolio or specific segments. This aggregation is typically done by summing up the individual ECLs.

This aggregated ECL figure is vital for financial reporting, capital allocation, and risk management. It informs decisions about loan pricing, portfolio diversification, and the adequacy of loss provisions.

### 7.3 The ECL Results

The calculated ECL values provide a forward-looking measure of credit risk. A higher ECL for a particular loan or segment indicates a greater anticipated loss due to default. These results can be used to:

*   **Inform Provisioning:** Banks set aside capital (provisions) to cover expected credit losses. The ECL calculation directly feeds into this process, ensuring that adequate reserves are maintained.
*   **Guide Lending Decisions:** Loans with very high ECL might be deemed too risky, or they might require higher interest rates to compensate for the increased risk.
*   **Portfolio Management:** By analyzing ECL across different segments (e.g., by credit grade, industry, or geographic region), institutions can identify concentrations of risk and take steps to mitigate them.
*   **Stress Testing:** ECL models can be used in stress testing scenarios to assess how credit losses would behave under adverse economic conditions.

In essence, the ECL calculation transforms the statistical predictions of default, loss, and exposure into a tangible financial figure, providing a comprehensive view of potential credit losses and enabling proactive risk management.


#### Explation of key Concepts


## 8. Independent Variables: Preprocessing and Feature Engineering

This section delves into the preparation of the independent variables, which are the features used to predict Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). The notebook employs a crucial technique called Weight of Evidence (WoE) and Information Value (IV) for feature engineering. This approach is particularly popular in credit risk modeling due to its ability to transform categorical and continuous variables into a format suitable for logistic regression, while also providing insights into their predictive power.

### 8.1 Understanding Weight of Evidence (WoE) and Information Value (IV)

**Weight of Evidence (WoE)** is a measure of the predictive power of an independent variable in relation to the dependent variable (in our case, default/non-default, recovery rate, or CCF). It quantifies how much the odds of the dependent variable change based on the different categories or bins of the independent variable. The formula for WoE for a given category is:

$WoE = \ln\left(\frac{\% of Non-Defaults in Category}{\% of Defaults in Category}\right)$

*   A positive WoE indicates that the category has a higher proportion of non-defaults compared to defaults, suggesting it's associated with lower risk.
*   A negative WoE indicates a higher proportion of defaults, suggesting higher risk.
*   WoE values are typically calculated for each bin of a continuous variable or each category of a categorical variable.

**Information Value (IV)** is a measure of the overall predictive power of an independent variable. It is calculated by summing up the product of the difference between the percentage of non-defaults and defaults in each category, and the WoE for that category. The formula for IV is:

$IV = \sum_{i=1}^{n} \left( \% of Non-Defaults_i - \% of Defaults_i \right) \times WoE_i$

Where $i$ represents each category or bin of the variable.

*   **IV < 0.02:** The variable is considered to have very weak predictive power.
*   **0.02 <= IV < 0.1:** Weak predictive power.
*   **0.1 <= IV < 0.3:** Medium predictive power.
*   **IV >= 0.3:** Strong predictive power.

WoE and IV are valuable for several reasons:

1.  **Handling Categorical Variables:** They effectively transform categorical variables into a continuous scale, making them suitable for linear models like logistic regression.
2.  **Handling Missing Values:** Missing values can be treated as a separate category, and their WoE can be calculated, allowing them to be incorporated into the model.
3.  **Monotonic Relationship:** WoE transformation often creates a monotonic relationship between the independent variable and the log-odds of the dependent variable, which is beneficial for model interpretability.
4.  **Feature Selection:** IV helps in identifying and selecting the most predictive features for the model, as variables with low IV can be discarded.

### 8.2 Preprocessing Steps in the Notebook

The notebook likely performs the following steps for each independent variable:

1.  **Binning Continuous Variables:** Continuous variables are often binned into a few discrete categories. This can be done using various methods, such as equal-frequency binning, equal-width binning, or custom binning based on business knowledge or statistical analysis (e.g., using decision trees to find optimal cut-off points).
2.  **Calculating WoE for Each Bin/Category:** For each bin or category of a variable, the WoE is calculated based on the proportion of defaults and non-defaults within that bin.
3.  **Calculating IV for Each Variable:** The IV is calculated for each variable to assess its overall predictive strength.
4.  **Replacing Original Values with WoE Values:** The original values of the independent variables are replaced with their corresponding WoE values. This transformed dataset is then used for model training.
5.  **Feature Selection based on IV:** Variables with IV below a certain threshold (e.g., 0.1) might be excluded from the model, as they contribute little to the predictive power.

This systematic approach ensures that the independent variables are well-prepared, highly predictive, and suitable for building robust credit risk models.




### 6.1 Preprocessing Independent Variables

Before diving into model building, the notebook meticulously prepares the independent variables. This is a crucial step as the quality of your input data directly impacts the reliability and accuracy of your models. The process involves several key transformations:

#### 6.1.1 Handling Missing Values

Missing data is a common challenge in real-world datasets. The notebook addresses this by replacing missing values in categorical variables with a placeholder '`_`'. For numerical variables, missing values are imputed with the mean of their respective columns. This approach ensures that all observations can be used in the modeling process, preventing data loss and potential biases.

#### 6.1.2 Feature Engineering: Creating Dummy Variables

Many of the independent variables are categorical, meaning they represent distinct categories rather than numerical quantities (e.g., 'Home Ownership' or 'Employment Status'). To make these variables usable by statistical models, they are converted into 'dummy' or 'indicator' variables. This involves creating a new binary column for each category within a categorical variable. For instance, if 'Home Ownership' has categories 'Rent', 'Own', and 'Mortgage', three new columns would be created: 'Home Ownership_Rent', 'Home Ownership_Own', and 'Home Ownership_Mortgage'. A '1' in one of these columns indicates the presence of that category, while a '0' indicates its absence. This transformation allows the models to interpret categorical information effectively.

#### 6.1.3 Data Splitting: Training and Testing Datasets

To rigorously evaluate the performance of the credit risk models, the dataset is split into two distinct parts: a training set and a testing set. The training set (typically 80% of the data) is used to 'teach' the models, allowing them to learn the relationships between the independent variables and the dependent variables (PD, LGD, EAD). The testing set (the remaining 20%) is then used to assess how well the trained models generalize to unseen data. This separation is vital for preventing 'overfitting,' a scenario where a model performs exceptionally well on the training data but poorly on new, real-world data. The notebook uses a random split, ensuring that both sets are representative of the overall dataset.

#### 6.1.4 Feature Scaling: Standardization

For many machine learning algorithms, especially those that rely on distance calculations (like logistic regression, which is often used for PD models), it's important to scale numerical features. The notebook employs standardization, which transforms the data so that it has a mean of 0 and a standard deviation of 1. This process, often called Z-score normalization, ensures that no single feature dominates the model simply because of its larger scale. For example, 'Annual Income' might have values in the tens of thousands, while 'Number of Dependents' might be single digits. Without scaling, 'Annual Income' could disproportionately influence the model. Standardization brings all numerical features to a comparable scale, leading to more stable and efficient model training.

### 6.2 Variable Selection and Model Building for Probability of Default (PD)

The core of any credit risk model lies in accurately predicting the probability of default. The notebook employs a meticulous process to select the most impactful variables and build a robust PD model.

#### 6.2.1 Initial Variable Screening: Chi-Squared Test

Before diving into complex modeling, the notebook performs an initial screening of categorical independent variables against the 'good_bad' dependent variable (which indicates default or non-default). The Chi-Squared test is used for this purpose. This statistical test helps determine if there's a significant association between a categorical independent variable and the 'good_bad' outcome. Variables with a high Chi-Squared statistic and a low p-value suggest a strong relationship, making them good candidates for inclusion in the model. This step helps in identifying potentially useful predictors early on.

#### 6.2.2 Information Value (IV) and Weight of Evidence (WOE)

For both categorical and numerical variables, the concepts of Information Value (IV) and Weight of Evidence (WOE) are employed. These are powerful techniques in credit risk modeling for variable selection and transformation:

*   **Weight of Evidence (WOE):** WOE measures the predictive power of a categorical variable (or binned numerical variable) in relation to the target variable (default/non-default). It quantifies how much the odds of default change for each category of a variable. A higher absolute WOE value indicates a stronger relationship. The notebook calculates WOE for each category of selected variables, transforming them into a continuous scale that can be directly used in logistic regression.

*   **Information Value (IV):** IV is a summary statistic derived from WOE. It measures the overall predictive power of a variable. Generally, IV values are interpreted as follows:
    *   Less than 0.02: Not useful for prediction
    *   0.02 to 0.1: Weak predictor
    *   0.1 to 0.3: Medium predictor
    *   0.3 to 0.5: Strong predictor
    *   Greater than 0.5: Suspiciously strong (might be overfitting or data leakage)

    The notebook uses IV to select variables that have meaningful predictive power, ensuring that only relevant features are fed into the PD model.

#### 6.2.3 Logistic Regression for PD Modeling

Logistic Regression is the workhorse for Probability of Default (PD) modeling due to its interpretability and effectiveness in binary classification problems. The notebook builds a logistic regression model using the selected and transformed independent variables to predict the likelihood of a borrower defaulting. The output of the logistic regression model is a probability score between 0 and 1, representing the estimated PD for each borrower.

#### 6.2.4 Model Evaluation: ROC Curve and AUC

After training the PD model, its performance is rigorously evaluated. The notebook utilizes the Receiver Operating Characteristic (ROC) curve and the Area Under the Curve (AUC) metric. These are standard tools for assessing the performance of binary classification models:

*   **ROC Curve:** The ROC curve plots the True Positive Rate (sensitivity) against the False Positive Rate (1-specificity) at various threshold settings. It visually represents the trade-off between correctly identifying defaulters and incorrectly classifying non-defaulters.

*   **AUC (Area Under the Curve):** The AUC quantifies the overall performance of the classifier. An AUC of 1 indicates a perfect model, while an AUC of 0.5 suggests a model no better than random guessing. A higher AUC value signifies a better-performing model. The notebook calculates and presents the AUC, providing a clear measure of the PD model's discriminatory power.

#### 6.2.5 Gini Coefficient and Kolmogorov-Smirnov (KS) Statistic

Further evaluation metrics are employed to assess the PD model's effectiveness:

*   **Gini Coefficient:** The Gini coefficient is derived from the AUC (Gini = 2 * AUC - 1). It measures the inequality among values of a frequency distribution. In credit risk, it indicates how well the model separates good and bad customers. A higher Gini coefficient implies better separation.

*   **Kolmogorov-Smirnov (KS) Statistic:** The KS statistic measures the maximum difference between the cumulative distribution functions of the 'good' and 'bad' customers. A higher KS value indicates a better ability of the model to distinguish between the two groups. It's particularly useful for finding the optimal cut-off point for classifying defaults.

### 6.3 Model Building for Loss Given Default (LGD)

Once the probability of default is estimated, the next crucial component for ECL calculation is the Loss Given Default (LGD). This represents the proportion of the exposure that is lost if a default occurs. The notebook models LGD using a linear regression approach.

#### 6.3.1 Data Preparation for LGD

For LGD modeling, the focus shifts to the subset of loans that have actually defaulted. The dependent variable for this model is the 'recovery rate', which is 1 minus the LGD. The notebook prepares this data by filtering for defaulted loans and ensuring the 'recovery rate' is within the logical bounds of 0 and 1. This often involves 'capping' or 'flooring' values that fall outside this range due to data anomalies.

#### 6.3.2 Linear Regression for LGD Modeling

Linear regression is used to model the recovery rate (and thus LGD). The independent variables used here are similar to those in the PD model, but their impact on recovery might differ. The model aims to predict the continuous value of the recovery rate based on the characteristics of the defaulted loans.

#### 6.3.3 Model Evaluation for LGD

For linear regression models, common evaluation metrics include:

*   **R-squared:** This statistic represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A higher R-squared indicates a better fit of the model to the data.

*   **Mean Squared Error (MSE) or Root Mean Squared Error (RMSE):** These metrics quantify the average magnitude of the errors between predicted and actual values. Lower values indicate better model accuracy.

### 6.4 Model Building for Exposure at Default (EAD)

The final component needed for Expected Credit Loss is the Exposure at Default (EAD). This is the outstanding amount a bank is exposed to when a borrower defaults. For revolving credit facilities (like credit cards or lines of credit), the EAD is not simply the current outstanding balance, as borrowers might draw down more funds before defaulting. The notebook models EAD using a linear regression approach.

#### 6.4.1 Data Preparation for EAD

Similar to LGD, EAD modeling focuses on defaulted accounts. The dependent variable is the 'EAD' itself, which is typically calculated as the outstanding balance at default plus any undrawn commitments that were utilized just before default. The notebook prepares this data, often involving specific calculations based on the type of credit product.

#### 6.4.2 Linear Regression for EAD Modeling

A linear regression model is built to predict the EAD based on relevant independent variables. These variables might include the original credit limit, current outstanding balance, and other factors that influence how much a borrower might draw down before defaulting.

#### 6.4.3 Model Evaluation for EAD

Evaluation metrics for the EAD linear regression model are similar to those for LGD, including R-squared and MSE/RMSE, to assess the model's accuracy in predicting the exposure at the time of default.





## 7. Probability of Default (PD) Model

This section delves into the heart of our credit risk assessment: the Probability of Default (PD) model. The PD model aims to predict the likelihood that a borrower will default on their financial obligations within a specified timeframe. This is a crucial component of the Expected Credit Loss (ECL) calculation, as a higher probability of default directly translates to higher expected losses.

### 7.1 Data Preparation for PD Model

Before we can build our PD model, we need to prepare the data. This involves selecting the relevant features (independent variables) that are most likely to influence a borrower's probability of default. The notebook carefully selects a subset of the preprocessed independent variables, ensuring that they are suitable for a logistic regression model. This step is critical because the quality of our input data directly impacts the accuracy and reliability of our model.

### 7.2 Building the Logistic Regression Model

Logistic regression is a statistical model that is widely used for binary classification problems, making it an excellent choice for predicting default (a binary outcome: default or no default). The model estimates the probability of a binary outcome by fitting data to a logistic function. In simpler terms, it learns the relationship between our chosen independent variables and the likelihood of default.

The notebook demonstrates the process of building this model using the `statsmodels` library in Python. This library provides robust tools for statistical modeling, allowing us to not only fit the model but also to examine its statistical properties in detail.

### 7.3 Interpreting the PD Model Results

Once the logistic regression model is built, the next crucial step is to interpret its results. The output from `statsmodels` provides a wealth of information, including:

*   **Coefficients:** These values indicate the strength and direction of the relationship between each independent variable and the log-odds of default. A positive coefficient suggests that an increase in the variable's value is associated with a higher probability of default, while a negative coefficient suggests the opposite.
*   **P-values:** These tell us the statistical significance of each coefficient. A low p-value (typically less than 0.05) indicates that the variable is statistically significant in predicting default, meaning its observed effect is unlikely to be due to random chance.
*   **Odds Ratios:** While not directly provided by `statsmodels` in its summary, these can be easily calculated from the coefficients (by exponentiating them). Odds ratios provide a more intuitive understanding of the impact of each variable. For example, an odds ratio of 1.5 for a particular variable means that for every one-unit increase in that variable, the odds of default increase by 50%, holding all other variables constant.
*   **Model Fit Statistics:** Metrics like the Log-Likelihood, AIC (Akaike Information Criterion), and BIC (Bayesian Information Criterion) help us assess the overall fit of the model. Lower values for AIC and BIC generally indicate a better-fitting model. The pseudo R-squared values (e.g., McFadden's R-squared) provide a measure of how well the model explains the variation in the dependent variable, similar to R-squared in linear regression, but interpreted differently.

The notebook meticulously walks through the interpretation of these outputs, highlighting which variables are significant predictors of default and the direction of their influence. This step is vital for understanding the drivers of credit risk and for validating the model's economic intuition.

### 7.4 Assessing Model Performance

Beyond statistical significance, it's essential to evaluate how well our PD model performs in practice. The notebook likely employs various techniques to assess performance, such as:

*   **Confusion Matrix:** This table summarizes the performance of a classification model by showing the counts of true positives, true negatives, false positives, and false negatives. It's a fundamental tool for understanding where the model is making correct and incorrect predictions.
*   **Accuracy, Precision, Recall, F1-score:** These metrics, derived from the confusion matrix, provide a quantitative measure of the model's predictive power. Accuracy tells us the proportion of correct predictions, precision focuses on the correctness of positive predictions, recall measures the model's ability to find all positive instances, and the F1-score is a harmonic mean of precision and recall.
*   **ROC Curve and AUC:** The Receiver Operating Characteristic (ROC) curve plots the true positive rate against the false positive rate at various threshold settings. The Area Under the Curve (AUC) provides a single scalar value that summarizes the overall performance of the classifier across all possible classification thresholds. An AUC of 1 indicates a perfect model, while an AUC of 0.5 suggests a model no better than random guessing.
*   **Gini Coefficient and KS Statistic:** These are commonly used in credit risk modeling to assess the discriminatory power of the model. The Gini coefficient is related to the AUC, and the KS (Kolmogorov-Smirnov) statistic measures the maximum difference between the cumulative true positive rate and cumulative false positive rate, indicating the model's ability to separate defaulters from non-defaulters.

The notebook's analysis of these performance metrics provides a clear picture of the PD model's strengths and weaknesses, guiding us on potential areas for improvement or further refinement.



### 7.5 PD Model Validation and Application

After building and interpreting the PD model, it's crucial to validate its robustness and predictive power. The notebook likely employs techniques such as out-of-sample testing, where the model's performance is assessed on a dataset it has not seen during training. This helps ensure that the model generalizes well to new, unseen data.

Furthermore, the model's stability over time and across different segments of the portfolio should be monitored. This involves back-testing and stress-testing the model to understand its behavior under various economic scenarios. The ultimate goal is to ensure that the PD model provides reliable and consistent estimates of default probabilities, which are essential for accurate ECL calculations and effective risk management.

## 8. Loss Given Default (LGD) Model

The Loss Given Default (LGD) model is another critical component of the Expected Credit Loss (ECL) framework. While the PD model estimates the likelihood of default, the LGD model quantifies the proportion of the exposure that will be lost if a default occurs. In essence, it tells us how much money we expect to lose on a defaulted loan after accounting for any recoveries.

### 8.1 Data Preparation for LGD Model

Similar to the PD model, the LGD model requires careful data preparation. The notebook focuses on the 'recovery_rate' variable, which represents the percentage of the funded amount that was recovered after a default. It's common practice to cap recovery rates at 1 (100%) and floor them at 0 (0%) to ensure they are within a meaningful range. The notebook also likely addresses any outliers or unusual distributions in the recovery rate data, as these can significantly impact the model's accuracy.

### 8.2 Building the LGD Model

The LGD model is typically built using regression techniques, as the 'recovery_rate' is a continuous variable. The notebook might explore different regression approaches, such as linear regression or more advanced techniques, to model the relationship between various independent variables and the recovery rate. The choice of model depends on the characteristics of the data and the desired level of complexity.

### 8.3 Interpreting the LGD Model Results

Interpreting the LGD model results involves analyzing the coefficients of the independent variables, their statistical significance (p-values), and the overall fit of the model (e.g., R-squared). The coefficients indicate how each variable influences the recovery rate. For example, a positive coefficient for collateral value would suggest that higher collateral is associated with higher recovery rates.

### 8.4 Assessing LGD Model Performance

Evaluating the LGD model's performance involves metrics such as Mean Squared Error (MSE) or Root Mean Squared Error (RMSE), which measure the average magnitude of the errors. Additionally, visual inspections of predicted versus actual recovery rates can help identify any systematic biases or areas where the model performs poorly. The goal is to ensure that the LGD model provides accurate and unbiased estimates of potential losses.

## 9. Exposure at Default (EAD) Model

The Exposure at Default (EAD) model estimates the outstanding amount of a loan or credit facility that is expected to be outstanding at the time of default. For committed but undrawn facilities, EAD is not simply the current outstanding balance but also includes a portion of the undrawn commitment. This is crucial for accurately assessing potential losses, as a borrower might draw down more of their available credit just before defaulting.

### 9.1 Data Preparation for EAD Model

The notebook calculates the Credit Conversion Factor (CCF), which is a key component of the EAD model. The CCF represents the percentage of the undrawn commitment that is expected to be drawn down by the time of default. The calculation of CCF involves the funded amount and the total principal recovered. The notebook likely preprocesses these variables to handle any missing values or outliers.

### 9.2 Building the EAD Model

The EAD model is typically built using regression techniques, similar to the LGD model, as the CCF is a continuous variable. The notebook might explore different regression approaches to model the relationship between various independent variables and the CCF. The choice of model depends on the characteristics of the data and the desired level of complexity.

### 9.3 Interpreting the EAD Model Results

Interpreting the EAD model results involves analyzing the coefficients of the independent variables, their statistical significance (p-values), and the overall fit of the model (e.g., R-squared). The coefficients indicate how each variable influences the CCF. For example, a positive coefficient for a variable like 'credit utilization' might suggest that borrowers with higher utilization rates are more likely to draw down their remaining credit before defaulting.

### 9.4 Assessing EAD Model Performance

Evaluating the EAD model's performance involves metrics such as Mean Squared Error (MSE) or Root Mean Squared Error (RMSE), which measure the average magnitude of the errors. Additionally, visual inspections of predicted versus actual CCF values can help identify any systematic biases or areas where the model performs poorly. The goal is to ensure that the EAD model provides accurate and unbiased estimates of potential exposures at the time of default.

## 10. Expected Credit Loss (ECL) Calculation

The Expected Credit Loss (ECL) is the ultimate output of our credit risk modeling framework. It represents the best estimate of the credit losses that are expected to arise over the lifetime of a financial instrument. The ECL is calculated by combining the outputs of the Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD) models.

### 10.1 The ECL Formula

The fundamental formula for Expected Credit Loss is:

**ECL = PD x LGD x EAD**

Where:
*   **PD (Probability of Default):** The likelihood that a borrower will default over a specific period.
*   **LGD (Loss Given Default):** The proportion of the exposure that will be lost if a default occurs.
*   **EAD (Exposure at Default):** The outstanding amount of the loan or credit facility at the time of default.

This formula is applied to each individual loan or credit facility in the portfolio to arrive at an aggregate ECL for the entire portfolio.

### 10.2 Calculating ECL in the Notebook

The notebook demonstrates how to combine the predicted values from the PD, LGD, and EAD models to calculate the ECL for each loan. This typically involves:

1.  **Predicting PD:** Using the trained PD model to generate a probability of default for each loan.
2.  **Predicting LGD:** Using the trained LGD model to estimate the loss given default for each loan.
3.  **Predicting EAD:** Using the trained EAD model to estimate the exposure at default for each loan.
4.  **Multiplying the Components:** Multiplying the predicted PD, LGD, and EAD for each loan to obtain its individual ECL.

### 10.3 Interpreting and Utilizing ECL

The calculated ECL is a crucial metric for financial institutions. It serves multiple purposes:

*   **Provisioning:** Financial institutions are required to set aside provisions (reserves) for expected credit losses. The ECL calculation directly informs the amount of these provisions, ensuring that potential losses are adequately covered.
*   **Risk Management:** ECL provides a forward-looking measure of credit risk, allowing institutions to proactively manage their portfolios, identify high-risk segments, and implement appropriate mitigation strategies.
*   **Capital Allocation:** By understanding the expected losses associated with different assets, institutions can make informed decisions about capital allocation, ensuring that capital is deployed efficiently and in line with risk appetite.
*   **Pricing:** ECL can be incorporated into the pricing of loans and other credit products, ensuring that the price adequately compensates for the expected credit risk.
*   **Regulatory Compliance:** Regulatory frameworks, such as IFRS 9 and CECL, mandate the calculation and reporting of ECL, making it a critical aspect of compliance.

The notebook's final ECL calculation provides a tangible output that can be used for these purposes, demonstrating the practical application of the credit risk models developed.

## Conclusion

This model documentation has provided a comprehensive overview of the credit risk model developed in the attached notebook. We have explored the key components of the Expected Credit Loss (ECL) framework: Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). For each component, we have detailed the data preparation, model building, interpretation of results, and performance assessment. The final ECL calculation integrates these components to provide a robust measure of expected credit losses.

This model serves as a valuable tool for financial institutions to assess, manage, and provision for credit risk effectively. By understanding the underlying methodologies and interpreting the model outputs, stakeholders can make informed decisions to ensure the financial health and stability of their portfolios. The detailed explanations provided aim to demystify the complex world of credit risk modeling, making it accessible to a broader audience while maintaining technical rigor.

