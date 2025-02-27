import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import pandas as pd
import numpy as np
import sys
import io
import os
import threading
from model_train import Model_train
from threshold import Threshold
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class RedirectOutput:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = io.StringIO()
        
    def write(self, string):
        self.buffer.write(string)
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)
        
    def flush(self):
        pass

class PredictiveMaintenanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sensor Anomaly Detection Tool")
        self.root.geometry("1200x700")
        self.root.configure(bg="#007bff")
        
        # Set styles
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#007bff")
        self.style.configure("Header.TFrame", background="#007bff")
        self.style.configure("White.TFrame", background="white")
        self.style.configure("Header.TLabel", background="#007bff", foreground="white", font=("Arial", 16, "bold"))
        self.style.configure("Info.TLabel", background="white", foreground="black")
        
        # Updated button styles with more modern look
        self.style.configure("TButton", background="#007bff", foreground="white", font=("Arial", 11, "bold"), padding=8)
        self.style.map("TButton", background=[("active", "#0069d9")], foreground=[("active", "white")])
        
        # Enhanced upload button style with updated font color
        self.style.configure("Upload.TButton", background="#28a745", foreground="#FFFF00", font=("Arial", 11, "bold"), padding=10)
        self.style.map("Upload.TButton", background=[("active", "#218838")], foreground=[("active", "#FFFF00")])
        
        # Enhanced train button style
        self.style.configure("Train.TButton", background="#ffc107", foreground="black", font=("Arial", 11, "bold"), padding=10)
        self.style.map("Train.TButton", background=[("active", "#e0a800")], foreground=[("active", "black")])
        
        # Configure the grid layout
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=0)  # Header
        self.root.rowconfigure(1, weight=1)  # Main content
        
        # Create the header frame
        self.create_header()
        
        # Create the main content frame
        self.create_main_content()
        
        # Initialize data variables
        self.df = None
        self.file_path = None
        self.monitoring_rules = {}
        self.thresholds = {}
        
    def create_header(self):
        header_frame = ttk.Frame(self.root, style="Header.TFrame", padding="10")
        header_frame.grid(row=0, column=0, sticky="ew")
        
        # Logo (using a simple label with an icon character)
        logo_label = ttk.Label(
            header_frame, 
            text="📊", 
            style="Header.TLabel",
            font=("Arial", 24)
        )
        logo_label.pack(side=tk.LEFT, padx=10)
        
        # Title
        title_label = ttk.Label(
            header_frame, 
            text="Sensor Anomaly Detection", 
            style="Header.TLabel"
        )
        title_label.pack(side=tk.LEFT, padx=10)
        
        # Upload button with improved style
        upload_button = ttk.Button(
            header_frame, 
            text="📂 Upload Dataset",  # Added icon
            command=self.upload_dataset,
            style="Upload.TButton"
        )
        upload_button.pack(side=tk.RIGHT, padx=10)
        
    def create_main_content(self):
        main_frame = ttk.Frame(self.root, padding="10", style="TFrame")
        main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        # Configure main frame grid
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel - Dataset Information
        info_frame = ttk.Frame(main_frame, style="White.TFrame")
        info_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # Dataset Information Header
        info_header = ttk.Label(
            info_frame, 
            text="Dataset Information",
            font=("Arial", 12, "bold"),
            background="#007bff",
            foreground="white",
            padding=10
        )
        info_header.pack(fill=tk.X)
        
        # File info container
        self.file_info_frame = ttk.Frame(info_frame, style="White.TFrame", padding="10")
        self.file_info_frame.pack(fill=tk.BOTH, expand=True)
        
        # File icon
        file_icon = ttk.Label(
            self.file_info_frame, 
            text="📄",
            font=("Arial", 24),
            style="Info.TLabel"
        )
        file_icon.grid(row=0, column=0, rowspan=3, padx=(0, 10))
        
        # File info labels
        self.file_name_label = ttk.Label(
            self.file_info_frame, 
            text="File: No file selected",
            style="Info.TLabel"
        )
        self.file_name_label.grid(row=0, column=1, sticky="w")
        
        self.file_rows_label = ttk.Label(
            self.file_info_frame, 
            text="Rows: -",
            style="Info.TLabel"
        )
        self.file_rows_label.grid(row=1, column=1, sticky="w")
        
        self.file_cols_label = ttk.Label(
            self.file_info_frame, 
            text="Columns: -",
            style="Info.TLabel"
        )
        self.file_cols_label.grid(row=2, column=1, sticky="w")
        
        self.numeric_cols_label = ttk.Label(
            self.file_info_frame, 
            text="Numeric columns: -",
            style="Info.TLabel"
        )
        self.numeric_cols_label.grid(row=3, column=1, sticky="w")
        
        # Control buttons
        control_frame = ttk.Frame(info_frame, style="White.TFrame", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        # Train model button with updated style
        self.train_button = ttk.Button(
            control_frame, 
            text="🔬 Train Model",  # Added icon
            command=self.train_model,
            state=tk.DISABLED,
            style="Train.TButton"  # Using the new yellow button style
        )
        self.train_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Right panel - Analysis Results
        results_frame = ttk.Frame(main_frame, style="White.TFrame")
        results_frame.grid(row=0, column=1, sticky="nsew")
        
        # Analysis Results Header
        results_header = ttk.Label(
            results_frame, 
            text="Analysis Results",
            font=("Arial", 12, "bold"),
            background="#007bff",
            foreground="white",
            padding=10
        )
        results_header.pack(fill=tk.X)
        
        # Tabs for different views
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Data Preview tab
        self.preview_tab = ttk.Frame(self.results_notebook, style="White.TFrame")
        self.results_notebook.add(self.preview_tab, text="Data Preview")
        
        # Create a frame to hold the preview
        self.preview_frame = ttk.Frame(self.preview_tab, style="White.TFrame", padding="10")
        self.preview_frame.pack(fill=tk.BOTH, expand=True)
        
        # Output tab
        self.output_tab = ttk.Frame(self.results_notebook, style="White.TFrame")
        self.results_notebook.add(self.output_tab, text="Detection Results")
        
        self.output_text = scrolledtext.ScrolledText(
            self.output_tab, 
            wrap=tk.WORD,
            bg="white",
            fg="black",
            font=("Consolas", 10)
        )
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.output_text.config(state=tk.DISABLED)
        
        # Redirect stdout to the text widget
        self.redirect = RedirectOutput(self.output_text)
        sys.stdout = self.redirect
        
        # Visualization tab
        self.viz_tab = ttk.Frame(self.results_notebook, style="White.TFrame")
        self.results_notebook.add(self.viz_tab, text="Visualization")
        
        # Create a canvas with scrollbar for scrolling visualizations
        viz_canvas_frame = ttk.Frame(self.viz_tab, style="White.TFrame")
        viz_canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Add scrollbar
        viz_scrollbar = ttk.Scrollbar(viz_canvas_frame, orient="vertical")
        viz_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create canvas
        viz_canvas = tk.Canvas(viz_canvas_frame, bg="white", yscrollcommand=viz_scrollbar.set)
        viz_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure scrollbar to control canvas
        viz_scrollbar.config(command=viz_canvas.yview)

        # Create frame inside canvas for content
        self.viz_container = ttk.Frame(viz_canvas, style="White.TFrame")
        viz_canvas.create_window((0, 0), window=self.viz_container, anchor="nw")

        # Configure function to update scroll region when size changes
        def viz_configure_scroll(event):
            viz_canvas.configure(scrollregion=viz_canvas.bbox("all"))
        self.viz_container.bind("<Configure>", viz_configure_scroll)
        
        self.viz_title = ttk.Label(
            self.viz_container,
            text="Sensor Anomaly Visualizations",
            font=("Arial", 12, "bold"),
            background="white",
            foreground="#007bff",
            padding=10
        )
        self.viz_title.pack(fill=tk.X)
        
    def upload_dataset(self):
        file_path = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Clear previous output
                self.output_text.config(state=tk.NORMAL)
                self.output_text.delete(1.0, tk.END)
                self.output_text.config(state=tk.DISABLED)
                
                self.file_path = file_path
                self.df = pd.read_csv(file_path)
                
                # Update file info
                file_name = os.path.basename(file_path)
                self.file_name_label.config(text=f"File: {file_name}")
                self.file_rows_label.config(text=f"Rows: {len(self.df)}")
                self.file_cols_label.config(text=f"Columns: {len(self.df.columns)}")
                
                # Count numeric columns
                numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
                self.numeric_cols_label.config(text=f"Numeric columns: {len(numeric_cols)}")
                
                # Create data preview
                self.create_data_preview()
                
                print(f"Dataset loaded: {file_name}")
                print(f"Shape: {self.df.shape}")
                print("Starting preprocessing automatically...")
                
                # Automatically start preprocessing
                threading.Thread(target=self._process_dataset_thread, daemon=True).start()
                
            except Exception as e:
                print(f"Error loading file: {str(e)}")
    
    def create_data_preview(self):
        # Clear previous preview
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        
        # Create a treeview to display the data
        columns = list(self.df.columns)
        
        # Create the treeview with scrollbars
        tree_frame = ttk.Frame(self.preview_frame, style="White.TFrame")
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        tree_scroll_y = ttk.Scrollbar(tree_frame, orient="vertical")
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        tree_scroll_x = ttk.Scrollbar(tree_frame, orient="horizontal")
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            yscrollcommand=tree_scroll_y.set,
            xscrollcommand=tree_scroll_x.set
        )
        
        # Configure the scrollbars
        tree_scroll_y.config(command=tree.yview)
        tree_scroll_x.config(command=tree.xview)
        
        # Set column headings
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        # Insert data (first 10 rows for performance)
        display_rows = min(10, len(self.df))
        for i in range(display_rows):
            values = self.df.iloc[i].tolist()
            # Convert any non-string values to strings
            values = [str(val) for val in values]
            tree.insert("", tk.END, values=values)
        
        tree.pack(fill=tk.BOTH, expand=True)
    
    def _process_dataset_thread(self):
        try:
            print("\n=== Starting Dataset Processing ===")
            
            # Run Threshold analysis on the dataset
            print("Running threshold analysis...")
            threshold_obj = Threshold(self.df)
            
            # Extract monitoring rules from the Threshold class output
            # Store the rules for visualization later
            self.extract_monitoring_rules()
            
            # Create visualizations for anomalous features
            self.create_visualizations_for_anomalous_features()
            
            # Enable train button after processing
            self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL))
            
            print("\n=== Threshold Analysis Complete ===")
            
        except Exception as e:
            print(f"Error processing dataset: {str(e)}")
    
    def extract_monitoring_rules(self):
        # This is a mock function to simulate extracting rules from the Threshold class
        # In a real application, you'd get these from your threshold object
        
        # Parse the last printed lines (monitoring rules) from the redirected output
        buffer_content = self.redirect.buffer.getvalue()
        lines = buffer_content.split('\n')
        
        rule_lines = []
        capture = False
        
        for line in lines:
            if "=== Predictions ===" in line:
                capture = True
                continue
            if capture and line.strip():
                rule_lines.append(line.strip())
        
        # Parse the rules into a dictionary
        self.monitoring_rules = {}
        for rule in rule_lines:
            try:
                parts = rule.split(":", 1)
                if len(parts) == 2:
                    failure_type = parts[0].strip()
                    condition = parts[1].strip()
                    
                    # Parse condition like "Temperature_C > 95.60"
                    condition_parts = condition.split()
                    if len(condition_parts) >= 3:
                        feature = condition_parts[0]
                        direction = condition_parts[1]
                        threshold = float(condition_parts[2])
                        
                        self.monitoring_rules[failure_type] = {
                            'feature': feature,
                            'direction': direction,
                            'threshold': threshold
                        }
            except:
                continue
        
        # If no rules were extracted (for example during testing), create some sample rules
        if not self.monitoring_rules:
            # Create rules for all numeric features
            numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()
            
            for col in numeric_columns:
                # Skip any obvious target columns or ID columns
                if col.lower() in ['target', 'label', 'class', 'failure', 'anomaly', 'id', 'machine_id', 'machineid']:
                    continue
                    
                # Create high threshold rule
                high_threshold = self.df[col].quantile(0.95)
                self.monitoring_rules[f'{col} (HIGH)'] = {
                    'feature': col,
                    'direction': '>',
                    'threshold': high_threshold
                }
                
                # Create low threshold rule
                low_threshold = self.df[col].quantile(0.05)
                self.monitoring_rules[f'{col} (LOW)'] = {
                    'feature': col,
                    'direction': '<',
                    'threshold': low_threshold
                }
    
    def create_visualizations_for_anomalous_features(self):
        # Clear previous visualizations
        for widget in self.viz_container.winfo_children():
            widget.destroy()
        
        # Get all numeric features for visualization
        numeric_features = self.df.select_dtypes(include=['number']).columns.tolist()
        
        # Filter out any ID columns, target columns or operating hours
        features_to_exclude = ['id', 'machine_id', 'machineid', 'machine', 'target', 'label', 'class', 'failure', 'anomaly', 
                              'operating_hours', 'hours', 'operation_hours', 'runtime', 'runtime_hours', 'operational_hours']
        features_to_visualize = []
        
        for col in numeric_features:
            # Skip features with common ID names or operating hours
            if any(exclude_term in col.lower() for exclude_term in features_to_exclude):
                continue
            
            # Only include features that have monitoring rules (anomalous behavior)
            has_rule = False
            for rule in self.monitoring_rules.values():
                if rule['feature'] == col:
                    has_rule = True
                    break
            
            if has_rule:
                features_to_visualize.append(col)
        
        if not features_to_visualize:
            # If no features have rules, show a message
            no_data_label = ttk.Label(
                self.viz_container,
                text="No anomalous features detected for visualization.",
                font=("Arial", 12),
                background="white",
                foreground="#007bff",
                padding=20
            )
            no_data_label.pack(fill=tk.BOTH, expand=True)
            return
        
        # Calculate number of rows needed (2 plots per row)
        row_count = (len(features_to_visualize) + 1) // 2  # Round up division
        
        # Create frames for each row
        plot_rows = []
        for i in range(row_count):
            row_frame = ttk.Frame(self.viz_container, style="White.TFrame")
            row_frame.pack(fill=tk.X, expand=True, pady=5)
            row_frame.columnconfigure(0, weight=1)
            row_frame.columnconfigure(1, weight=1)
            plot_rows.append(row_frame)
        
        # Create plots for anomalous features
        for i, feature in enumerate(features_to_visualize):
            row = i // 2
            col = i % 2
            
            # Find associated rules for this feature
            high_rule = None
            low_rule = None
            
            for rule_name, rule in self.monitoring_rules.items():
                if rule['feature'] == feature:
                    if rule['direction'] == '>':
                        high_rule = rule
                    elif rule['direction'] == '<':
                        low_rule = rule
            
            # Create plot for this feature
            self.create_feature_plot(plot_rows[row], col, feature, high_rule, low_rule)
    
    def create_feature_plot(self, parent_frame, col, feature, high_rule=None, low_rule=None):
        # Get data for this feature
        if feature not in self.df.columns:
            return
            
        data = self.df[feature].values
        
        # Create a figure for the plot
        fig = Figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot all data points
        x = np.arange(len(data))
        ax.plot(x, data, color='#5B9BD5', label='Sensor readings')  # Milder blue
        
        # Track anomalies
        high_anomalies = np.zeros(len(data), dtype=bool)
        low_anomalies = np.zeros(len(data), dtype=bool)
        
        # Add high threshold if present
        if high_rule:
            high_threshold = high_rule['threshold']
            ax.axhline(y=high_threshold, color='#E57373', linestyle='-', label=f'High Threshold ({high_threshold:.2f})')  # Milder red
            high_anomalies = data > high_threshold
            
        # Add low threshold if present
        if low_rule:
            low_threshold = low_rule['threshold']
            ax.axhline(y=low_threshold, color='#FFB74D', linestyle='-', label=f'Low Threshold ({low_threshold:.2f})')  # Milder orange
            low_anomalies = data < low_threshold
        
        # Highlight anomalies
        high_anomaly_count = np.sum(high_anomalies)
        if high_anomaly_count > 0:
            ax.scatter(x[high_anomalies], data[high_anomalies], color='#D32F2F', s=30, label=f'High Anomalies ({high_anomaly_count})')  # Milder but visible red
        
        low_anomaly_count = np.sum(low_anomalies)
        if low_anomaly_count > 0:
            ax.scatter(x[low_anomalies], data[low_anomalies], color='#F57C00', s=30, label=f'Low Anomalies ({low_anomaly_count})')  # Milder but visible orange
        
        # Set labels and title
        anomaly_count = high_anomaly_count + low_anomaly_count
        ax.set_title(f"{feature} - {anomaly_count} anomalies detected")
        ax.set_xlabel("Reading Index")
        ax.set_ylabel(feature)
        
        # Add legend
        ax.legend(loc='best', fontsize='small')
        
        # Tight layout
        fig.tight_layout()
        
        # Create a container for the plot
        plot_frame = ttk.Frame(parent_frame, style="White.TFrame")
        plot_frame.grid(row=0, column=col, sticky="nsew", padx=5)
        
        # Add the plot to the container
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def train_model(self):
        if self.df is None:
            print("Please upload a dataset first.")
            return
        
        # Disable buttons during training
        self.train_button.config(state=tk.DISABLED)
        
        # Run training in a separate thread
        threading.Thread(target=self._train_model_thread, daemon=True).start()
    
    def _train_model_thread(self):
        try:
            print("\n=== Starting Model Training ===")
            
            # Prepare features and target
            features = list(self.df.columns)
            X = self.df.drop([features[-1]], axis=1)
            y = self.df[features[-1]]
            
            # Train model
            print("Training model (this may take some time)...")
            model = Model_train(X, y)
            
            print("\n=== Model Training Complete ===")
            print("Model has been trained and saved.")
            
            # Example prediction
            columns = list(X.columns)
            print("\nExample prediction with sample data:")
            data = list(X.iloc[0].values)  # Use first row as example
            print(f"Sample data: {data}")
            
            model.predict(data, columns)
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
        finally:
            # Re-enable train button
            self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL))

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictiveMaintenanceApp(root)
    root.mainloop()