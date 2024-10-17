import customtkinter as ctk
import joblib
import pandas as pd
# Load the saved model, scaler, and feature names
model = joblib.load('cardio_model.joblib')
scaler = joblib.load('cardio_scaler.joblib')
feature_names = joblib.load('feature_names.joblib')

class CardiovascularDiseaseApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Cardiovascular Disease Predictor")
        self.geometry("400x700")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame = ctk.CTkScrollableFrame(self)
        self.frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.title_label = ctk.CTkLabel(self.frame, text="Enter Patient Information", font=("Arial", 18, "bold"))
        self.title_label.grid(row=0, column=0, columnspan=2, pady=10)

        self.entries = {}
        self.create_entry("Age (years)", "age")
        self.create_entry("Gender (1:female, 2:male)", "gender")
        self.create_entry("Height (cm)", "height")
        self.create_entry("Weight (kg)", "weight")
        self.create_entry("Systolic BP", "ap_hi")
        self.create_entry("Diastolic BP", "ap_lo")
        self.create_entry("Cholesterol (1:normal, 2:above normal, 3:well above normal)", "cholesterol")
        self.create_entry("Glucose (1:normal, 2:above normal, 3:well above normal)", "gluc")
        self.create_entry("Smoking (0:no, 1:yes)", "smoke")
        self.create_entry("Alcohol intake (0:no, 1:yes)", "alco")
        self.create_entry("Physical activity (0:no, 1:yes)", "active")

        self.predict_button = ctk.CTkButton(self.frame, text="Predict", command=self.predict)
        self.predict_button.grid(row=len(self.entries)+1, column=0, columnspan=2, pady=10)

        self.result_label = ctk.CTkLabel(self.frame, text="", font=("Arial", 16))
        self.result_label.grid(row=len(self.entries)+2, column=0, columnspan=2, pady=10)

    def create_entry(self, label, key):
        row = len(self.entries) + 1
        ctk.CTkLabel(self.frame, text=label).grid(row=row, column=0, sticky="e", padx=5, pady=2)
        self.entries[key] = ctk.CTkEntry(self.frame)
        self.entries[key].grid(row=row, column=1, padx=5, pady=2)

    def predict(self):
        try:
            # Create a dictionary to hold the input values
            input_dict = {key: float(self.entries[key].get()) for key in self.entries}
            
            # Create a DataFrame with the correct feature names
            input_df = pd.DataFrame([input_dict])
            
            # One-hot encode categorical variables
            input_df = pd.get_dummies(input_df, columns=['cholesterol', 'gluc'])
            
            # Ensure all columns from training are present
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            # Reorder columns to match the training data
            input_df = input_df[feature_names]
            
            # Scale the input
            scaled_input = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(scaled_input)
            
            result = "High risk of cardiovascular disease" if prediction[0] == 1 else "Low risk of cardiovascular disease"
            self.result_label.configure(text=result)
        except ValueError:
            self.result_label.configure(text="Please enter valid numeric values")
        except Exception as e:
            self.result_label.configure(text=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app = CardiovascularDiseaseApp()
    app.mainloop()