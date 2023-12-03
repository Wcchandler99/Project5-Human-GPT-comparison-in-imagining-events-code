import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from supervised.automl import AutoML
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"~/Documents/Topics_in_AI/Final/ML_final_data.csv")

X = df[["story_sequentiality", "story_average_sentence_length", "story_concreteness_mean", "story_concreteness_lower", "story_concreteness_upper", "story_sentiment_mean"]]
y = df["is_gpt"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Train
automl = AutoML(algorithms=["Decision Tree", "Linear", "Random Forest"], total_time_limit=5*60)
automl.fit(X_train, y_train)

# Test
y_predicted = automl.predict(X_test)

predicted_data = pd.DataFrame({"Predicted": y_predicted, "Target": np.array(y_test)})

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predicted)
print(f"Accuracy: {accuracy}")

# Printing the confusion matrix
conf_matrix = confusion_matrix(y_test, y_predicted)
print("Confusion Matrix:")
print(conf_matrix)

# Creating a heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

X_train_sm = sm.add_constant(X_train)

# Fitting the best model using statsmodels
model_sm = sm.Logit(y_train, X_train_sm)
result = model_sm.fit()

# Print the summary to see p-values
print(result.summary())