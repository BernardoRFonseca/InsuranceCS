# InsuranceCS
## 
![is-it-better-to-cancel-an-unused-credit-card-or-keep-it-960x450](https://user-images.githubusercontent.com/68538809/142218390-a331bfc7-deb6-44c2-99c0-a8a74ad0ac24.png)


# 1. Business Context.

The company is a health insurance provider. The product team is looking into introducing a new product to their clients: vehicle insurance.

Vehicle Insurance (also known as motor insurance) intends to provide financial protection against physical damage or bodily injury resulting from traffic collisions and against liability that could arise from incidents in a vehicle.
Around 380,000 customers were surveyed to find out if they had an interest in joining a new vehicle insurance product. To the customers attributes database was added the interest feature.

Outside of the 380,000 clients scope, the product team selected 127,000 customers that were not surveyed to be offered the new vehicle insurance product.

Phone calls will be used to offer the new product. The sales team has a budget limit of 20,000 calls through the campaign period.

In this context, this project intends to provide the sales team with a solution that ranks the 127,000 by likelihood of purchasing the new insurance to decide which 20,000 potential customers should be contacted and by what order.

## 1.1 Project Deliverables.

In this business context, as a data scientist, I intend to develop a machine learning model that ranks customers by the likelihood of purchasing a new product.
 
With this model, the products team expects to prioritize clients that would be interested in the new product and so optimize sales campaigns by phone calling as much interested clients as possible.

A report with the following analysis is the deliverable goals:

1. Key findings on interested customers most relevant atributes.

2. How many interest costumers will the sales team be able to reach with 20,000 phone calls, and how much does that represent of the overall interested costumers.

3. How many interest costumers would the sales team be able to reach if the budget was increased to 40,000 calls, and how much that represents of the overall interested costumers.

# 2. The Solution

## Solution strategy

The following strategy was used to achieve the goal:

### Step 1. Data Description

The initial dataset has 381.109 rows and 12 columns. Follows the features description:

- **Id**: Unique ID for the customer   
- **Gender**: Gender of the customer 
- **Age**: Age of the customer  
- **Driving License**: 0, customer does not have DL; 1, customer already has DL  
- **Region Code**: Unique code for the region of the customer 
- **Previously Insured**: 1, customer already has vehicle insurance; 0, customer doesn't have vehicle insurance
- **Vehicle Age**: Age of the vehicle 
- **Vehicle Damage**: 1, customer got his/her vehicle damaged in the past; 0, customer didn't get his/her vehicle damaged in the past
- **Anual Premium**: The amount customer needs to pay as premium in the year 
- **Policy sales channel**: Anonymized Code for the channel of outreaching to the customer ie 
- **Vintage**: Number of Days, customer has been associated with the company 
- **Response**: 1, customer is interested; 0, customer is not interested.  

Numerical attributes statistic analysis:

 ![numericalfeatures](https://user-images.githubusercontent.com/68538809/144523197-f12e4d30-6fe9-4cd1-b995-14c54d0728fa.JPG)

Categorical attributes statistic analysis:

![categorical features](https://user-images.githubusercontent.com/68538809/144523226-d0f0728e-1a8c-43be-b7d0-a3fdb3879800.JPG)

### Step 2. Feature Engineering

On feature creation, 2 columns were modified.
'Vehicle Change' values were changed from categorical to numerical.
'Vehicle Age' values were changes to a standardized descriptive format.

### Step 3. Exploratory Data Analysis (EDA)

On Exploratory Data Analysis, Univariate, Bivariate and Multivariates study was performed to help understand the statistical properties of each attributes, correlations and hypothesis testing.

### Step 4. Data Preparation

On this section, Standardization (StandardScaler Scaler), Rescaling (MinMax Scaler) and Encoding Transformations (Target Encoder) of the variables was carried out.

### Step 5. Feature Selection

To select the features to be used, two methods were used:

1. Application of Feature Importance using ExtraTreesClassifier method;
2. EDA insights .

**From Feature Importance:** 'vintage', 'annual_premium', 'age', 'region_code', 'vehicle_damage' and 'policy_sales_channel'

**From EDA:** 'driving_license'

### Step 6. Machine Learning Modelling

Machine learning models were trained and passed through Cross-Validation to evaluate the "real" performance. 

### Step 7. Hyperparameter Fine Tuning

Based on Machine Learning Modelling results, the best model was chosen and submited to Hyperparameter Fine Tuning to optize its performance.

### Step 8. Performance Evaluation and Interpretation

One of the main steps is the performance evaluation and interpertration, this stage intends to translate the model performance into business value. In this case, a correct classification of customers intention will bring a higher volume of sales inside a limited investment budget.

# 3. Data Insights

# 4. Machine Learning Model Applied

The following Machine Learning model were tested and cross-validated:

- **Logistic Regression**
- **Naive Bayes**
- **Extra Trees**
- **Random Forest Regressor**
- **K-Nearest Neighbors Regressor (KNN)**
- **XGBoost Classifier**
- **LightGBM Classifier**
- **CatBoost Classifier**

# 5. Machine Learning Model Performance

To evaluate the performance of the models, 4 metrics were used:

**ROC AUC**

**Precision at k**

**Recall at k**

**F1_Score**

The indicator "Recall at K" was the one to assume the major importance as it represents the proportion of relevante items found within the total number of defined item.

The following table discloses the cross-validated ("real") performance of all the models used.
**LGBM Classifier"** the best Recall at k result, making it the one be used.

![CVResults]((https://user-images.githubusercontent.com/68538809/147411461-cb319f17-615f-4ea6-af85-9b0c1870a19f.JPG)

Performance plots show the ability of the modelling identifying **all customers interested** in purchasing the product using approximatily **50% of the sample** (Cumulative Gain Curve). It is also possible to see that the model is initially **more than 3 times better than a random selection** (Lift Curve). It is important to notice that even after reaching half the sample, the model remains to perform 2 times better than a random method. 

![PerformancePlots](https://user-images.githubusercontent.com/68538809/147411463-07396ba0-0506-4dbd-bcb8-4a8d30cb4acc.png)

# 6. Business Results

**1.** Key findings on interested customers most relevant atributes.

**2.** How many interest costumers will the sales team be able to reach with 20,000 phone calls, and how much does that represent of the overall interested costumers.

With 20,000 calls, the sales team will be able to **reach 69.24%** of the interested customers. 

![Plot20000](https://user-images.githubusercontent.com/68538809/147411466-1925b693-ea2d-467f-8e0f-dd37dcff9283.JPG)

**3.** How many interest costumers would the sales team be able to reach if the budget was increased to 40,000 calls, and how much that represents of the overall interested costumers.

With 40,000 calls, the sales team will be able to **reach 98.64%** of the interested customers.

![Plot40000](https://user-images.githubusercontent.com/68538809/147411469-80b1ef88-506e-429f-b474-c6cab8ef0874.JPG)


# 7. Conclusions

# 8. Next Steps to Improve

