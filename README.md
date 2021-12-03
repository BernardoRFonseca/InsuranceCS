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

4. How many calls would the team need to make to achieve 85% of the customers interested in purchasing the motor insurance.


# 2. The Solution

## Solution strategy

The following strategy was used to achieve the goal:

### Step 1. Data Description

The initial dataset has 381.109 rows and 12 columns. Follows the features description:

![datadictionary](https://user-images.githubusercontent.com/68538809/144522779-112488d8-f1e3-43d9-b3e3-bae4784c1f6a.JPG)

Numerical attributes statistic analysis:

 ![numericalfeatures](https://user-images.githubusercontent.com/68538809/144523197-f12e4d30-6fe9-4cd1-b995-14c54d0728fa.JPG)

Categorical attributed statistic analysis:

![categorical features](https://user-images.githubusercontent.com/68538809/144523226-d0f0728e-1a8c-43be-b7d0-a3fdb3879800.JPG)

### Step 2. Feature Engineering

On feature creation, 2 columns were modified.
'Vehicle Change' values were changed from categorical to numerical.
'Vehicle Age' values were changes to a standardized descriptive format.

### Step 3. Exploratory Data Analysis (EDA)

### Step 4. Data Preparation

### Step 5. Feature Selection

### Step 6. Machine Learning Modelling

### Step 7. Hyperparameter Fine Tuning

### Step 8. Performance Evaluation and Interpretation


# 3. Data Insights

# 4. Machine Learning Model Applied

# 5. Machine Learning Model Performance

# 6. Business Results

# 7. Conclusions

# 8. Next Steps to Improve

