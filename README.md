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

### 5.2 Stage 2: Predicting Recovery Rate for Non-Zero Recoveries

For loans where we predicted a non-zero recovery (i.e., `recovery_rate_0_1` is 1), we built a linear regression model to predict the actual `recovery_rate`.

#### 5.2.1 Data Preparation for Stage 2 LGD Model

We filter the `loan_data_defaults` to include only those with non-zero recovery and then prepare the features and target variable.

#### 5.2.2 Splitting Data for Stage 2 LGD Model

Again, we split the data into training and testing sets for proper evaluation of the linear regression model.

- This performs the same splitting logic as in Stage 1, but on the `loan_data_defaults_nonzero_recovery` dataset.

#### 5.2.3 Training the Linear Regression Model

We use `LinearRegression` from `sklearn.linear_model` to predict the continuous `recovery_rate`.


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


## Conclusion

We have explored the key components of the Expected Credit Loss (ECL) framework: Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). The final ECL calculation integrates these components to provide a robust measure of expected credit losses.

This model serves as a valuable tool for financial institutions to assess, manage, and provision for credit risk effectively. By understanding the underlying methodologies and interpreting the model outputs, stakeholders can make informed decisions to ensure the financial health and stability of their portfolios.

