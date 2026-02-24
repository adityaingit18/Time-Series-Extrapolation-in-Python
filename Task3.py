import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Loading the data

df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# Listing in factors (columns) which will be required for the training of the prediction model that if a person will default or not

factors = ["credit_lines_outstanding","loan_amt_outstanding","total_debt_outstanding","income","years_employed","fico_score"]
X = df[factors]
y = df["default"]

# I will be using 80% of the given data to train the model
# I will be using 20% of the given data to test it's accuracy

X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size = 0.2, random_state=67)

# I will scale the data such that big figures like income would become comparable to small values like in fico_score

scaling_data = StandardScaler()
X_training_scaled = scaling_data.fit_transform(X_training)
X_testing_scaled = scaling_data.transform(X_testing)

# building a logistic regression model

logi_reg_model = LogisticRegression()
logi_reg_model.fit(X_training_scaled,y_training)
logi_reg_prob = logi_reg_model.predict_proba(X_testing_scaled)[:,1]

# building a random forest model

rand_for_model = RandomForestClassifier(n_estimators=100, random_state=67)
rand_for_model.fit(X_training_scaled,y_training)
rand_for_prob = rand_for_model.predict_proba(X_testing_scaled)[:,1]

print("| Comparative Analysis with 2 Different Models |")
print(f"Logistic Regression Accuracy: {roc_auc_score(y_testing, logi_reg_prob):.4f}")
print(f"Random Forest Accuracy: {roc_auc_score(y_testing, rand_for_prob):.4f}")

# After running the code, as I noticed that accuracy of Logistic Regression approach is better hence i will be moving forward with that approach

finalised_model = logi_reg_model

# now i will build an expected loss calculator function

def calc_expected_loss(credit_lines,loan_amt,total_debt,income,years_employed,fico_score):
    # first i will try to find probability of default
    data_applicant = pd.DataFrame([[credit_lines,loan_amt,total_debt,income,years_employed,fico_score]], columns = factors)
    scale_applicant_data = scaling_data.transform(data_applicant) # Scaling the data same way we did earlier
    prob_default_value = finalised_model.predict_proba(scale_applicant_data)[0][1] # Prediction of Defaulting Probability
    expected_loss = loan_amt * prob_default_value * 0.9 # As recovery rate is 10%, therefore loss given the default = 90% / 0.9

    print("\n| Loan Application Report |")
    print(f"Applicant's FICO Score: {fico_score} | Income: ${income:,.0f} | Employed for {years_employed} years.")
    print(f"Loan Amount Requested: ${loan_amt:,.2f}")
    print(f"Probability of Default (PD): {prob_default_value:.2%}")
    print(f"Expected Loss (in $): ${expected_loss:,.2f}")

    return expected_loss

# building an input function for user
while True:
    try:
        print("\n")
        user_input = input("Enter your FICO Score (e.g., 650) or 'quit': ")
        if user_input.lower() in ["quit",'q','exit']:
            print("\nClosing Loan Evaluator, Thank you for using!")
            break
        fico_score = int(user_input)
        credit_lines = int(input("Enter outstanding credit lines: "))
        loan_amt = float(input("Your loan amount you wish to request ($): "))
        total_debt = float(input("Your current total debt ($): "))
        income = float(input("Your annual income ($): "))
        years_employed = int(input("Years you've been employed for: "))

        calc_expected_loss(credit_lines=credit_lines,loan_amt=loan_amt,total_debt=total_debt,income=income,years_employed=years_employed,fico_score=fico_score)
    
    except ValueError:
        print("\nInvalid Input! Please enter numerical values only. Start Over!")