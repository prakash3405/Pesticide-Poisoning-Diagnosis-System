import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# Load the newly uploaded dataset
file_path = "updated_pesticide_poisoning_data.csv"
df = pd.read_csv(file_path)

# Display the first few rows to understand the data structure
df.head()
# Encode categorical features using Label Encoding
label_encoders = {}
categorical_columns = ['Gender', 'Pesticide_Type', 'Location', 'Symptoms', 'Pesticide_Contact']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target variable
X = df.drop(columns=['Label'])
y = df['Label']

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models
svc = SVC()
log_reg = LogisticRegression()
gbc = GradientBoostingClassifier()

# Voting Classifier combining the models
voting_clf = VotingClassifier(estimators=[('SVC', svc), ('LogReg', log_reg), ('GBC', gbc)], voting='hard')

# Train the models
svc.fit(X_train, y_train)
log_reg.fit(X_train, y_train)
gbc.fit(X_train, y_train)
voting_clf.fit(X_train, y_train)

# Evaluate model accuracies
svc_acc = accuracy_score(y_test, svc.predict(X_test))
log_reg_acc = accuracy_score(y_test, log_reg.predict(X_test))
gbc_acc = accuracy_score(y_test, gbc.predict(X_test))
voting_acc = accuracy_score(y_test, voting_clf.predict(X_test))

print(svc_acc, log_reg_acc, gbc_acc, voting_acc)
# Given input for prediction
input_data = [34, 10, 5, 1, 8, 1, 'Male', 'Fungicides', 'Narayanpet', 'Vomiting', 'Direct']

# Encode categorical values using stored label encoders
encoded_input = input_data[:6]  # First six values are numeric
for i, col in enumerate(categorical_columns):
    encoded_input.append(label_encoders[col].transform([input_data[6 + i]])[0])

# Scale input data
scaled_input = scaler.transform([encoded_input])

# Predict using the best model (Voting Classifier)
prediction = voting_clf.predict(scaled_input)[0]

# Decode prediction if it's categorical
print( prediction)
