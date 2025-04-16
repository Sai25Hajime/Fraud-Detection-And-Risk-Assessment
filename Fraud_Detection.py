import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Data Preprocessing
def preprocess_data(file_path='transaction_data.csv'):
    df = pd.read_csv(file_path)
    X = df[['amount', 'time_of_day']]
    y = df['is_fraud']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler, df  # Return df for visualization

# Train XGBoost Model
def train_model(X_train, X_test, y_train, y_test):
    model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Model Accuracy on Test Set: {accuracy:.4f}")
    return model

# Visualization Functions
def create_bubble_graph(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(df['time_of_day'], df['amount'], 
                        s=df['amount']*0.1,  # Bubble size based on amount
                        c=df['is_fraud'], cmap='RdYlGn', alpha=0.6)
    ax.set_xlabel('Time of Day (0-23)')
    ax.set_ylabel('Transaction Amount ($)')
    ax.set_title('Bubble Graph: Transactions (Red=Fraud, Green=Legit)')
    plt.colorbar(scatter, label='Is Fraud')
    return fig

def create_run_chart(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df['transaction_id'], df['amount'], label='Amount', color='blue')
    ax.scatter(df[df['is_fraud'] == 1]['transaction_id'], 
              df[df['is_fraud'] == 1]['amount'], 
              color='red', label='Fraudulent', zorder=5)
    ax.set_xlabel('Transaction ID')
    ax.set_ylabel('Amount ($)')
    ax.set_title('Run Chart: Transaction Amounts Over Sequence')
    ax.legend()
    return fig
    
# GUI Class
class FraudRiskApp:
    def __init__(self, root, model, scaler, df):
        self.root = root
        self.root.title("Fraud Detection & Risk Assessment")
        self.root.geometry("800x600")  # Increased size for visualizations
        
        self.model = model
        self.scaler = scaler
        self.df = df
        
        # Main Frame
        main_frame = ttk.Frame(root)
        main_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        # Input Frame
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(side="top", pady=10)
        
        ttk.Label(input_frame, text="Fraud & Risk Assessment", font=("Helvetica", 16, "bold")).pack(pady=10)
        
        ttk.Label(input_frame, text="Transaction Amount ($):").pack(pady=5)
        self.amount_entry = ttk.Entry(input_frame)
        self.amount_entry.pack(pady=5)
        
        ttk.Label(input_frame, text="Time of Day (0-23):").pack(pady=5)
        self.time_entry = ttk.Entry(input_frame)
        self.time_entry.pack(pady=5)
        
        ttk.Button(input_frame, text="Assess Transaction", command=self.assess).pack(pady=10)
        
        self.result_label = ttk.Label(input_frame, text="")
        self.result_label.pack(pady=5)
        self.risk_label = ttk.Label(input_frame, text="")
        self.risk_label.pack(pady=5)
        
        # Visualization Frame
        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(side="top", pady=10, fill="both", expand=True)
        
        # Bubble Graph
        bubble_fig = create_bubble_graph(self.df)
        bubble_canvas = FigureCanvasTkAgg(bubble_fig, master=viz_frame)
        bubble_canvas.draw()
        bubble_canvas.get_tk_widget().pack(side="left", padx=5, fill="both", expand=True)
        
        # Run Chart
        run_fig = create_run_chart(self.df)
        run_canvas = FigureCanvasTkAgg(run_fig, master=viz_frame)
        run_canvas.draw()
        run_canvas.get_tk_widget().pack(side="right", padx=5, fill="both", expand=True)

        # Quit Button
        ttk.Button(input_frame, text="Quit", command=self.quit_tk).pack(padx=2, pady=2)
    
    def assess(self):
        try:
            amount = float(self.amount_entry.get())
            time = int(self.time_entry.get())
            
            if amount < 0:
                messagebox.showerror("Error", "Amount cannot be negative")
                return
            if time < 0 or time > 23:
                messagebox.showerror("Error", "Time must be between 0 and 23")
                return
            
            input_data = np.array([[amount, time]])
            input_scaled = self.scaler.transform(input_data)
            
            fraud_pred = self.model.predict(input_scaled)[0]
            fraud_prob = self.model.predict_proba(input_scaled)[0][1]
            
            result = "FRAUD DETECTED" if fraud_pred == 1 else "Transaction is LEGIT"
            color = "red" if fraud_pred == 1 else "green"
            self.result_label.config(text=f"Prediction: {result}", foreground=color)
            
            risk_score = int(fraud_prob * 100)
            risk_color = "red" if risk_score > 75 else "orange" if risk_score > 25 else "green"
            self.risk_label.config(text=f"Risk Score: {risk_score}%", foreground=risk_color)
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def quit_tk(self):
        self.root.destroy()
        sys.exit()

# Main Execution
if __name__ == "__main__":
    # Preprocess and train
    X_train, X_test, y_train, y_test, scaler, df = preprocess_data()
    model = train_model(X_train, X_test, y_train, y_test)
    
    # Launch GUI
    root = tk.Tk()
    app = FraudRiskApp(root, model, scaler, df)
    root.mainloop()
