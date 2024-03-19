import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import dump

# Load the dataset
data = pd.read_csv("hand_landmarks_data.csv")

# Split features and labels
X = data.drop('label', axis=1)  # Assuming 'label' column contains the target labels (A or B)
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train, y_train)

# Predict labels for the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the trained model
model_file = "full_alphabet_scaled_ratio_based_svm_hand_gesture_classifier.joblib"
dump(svm_classifier, model_file)
print("Trained model saved as", model_file)
