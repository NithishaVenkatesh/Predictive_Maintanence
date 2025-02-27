import pandas as pd
import google.generativeai as genai
import ast
import pandas as pd
import numpy as np

# Load dataset

class Threshold:
    def __init__(self, df):
    # df = pd.read_csv('predictive_maintenance_large.csv')
    # Extract features and classes
        feature = list(df.columns)
        class_name = list(df[feature[-1]].unique())

        # Configure API key securely
        api_key = "AIzaSyAzQvavT2bKTdH2b2inWKS3WueS_vFG9cw"  # Replace with your API key securely
        genai.configure(api_key=api_key)

        # Define the Gemini model
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=[
                "You are supposed to find the exact feature which is highly related to the class "
                "and give the output in a dictionary format. Also, exclude the useless class which has no feature."
                "You must return the response strictly in valid JSON format without any extra explanations or formatting."

            ]
        )

        # Define user input
        user_input = f"features= {feature}, class={class_name} Do NOT include explanations, comments, or formatting beyond this JSON structure."

        # Generate response
        response = model.generate_content(user_input)
        # Print response from Gemini




        texts = response.text
        texts = texts.replace("```", "").replace("json", "").replace("{", "").replace("}", "").strip()

        # Manually reconstruct the dictionary format
        texts = "{" + texts + "}"

        # Convert to dictionary safely
        fail_feature_dict = ast.literal_eval(texts)


        def find_specific_failure_thresholds(df):
            """
            Find threshold values only for relevant feature-failure combinations.
            Maps each failure type to its most relevant feature.
            """

            # Define the mapping of failure types to their relevant features
            failure_feature_mapping = fail_feature_dict


            # Dictionary to store thresholds
            thresholds = {}

            # Process each failure type and its relevant feature
            for failure_type, feature in failure_feature_mapping.items():
                if feature not in df.columns:
                    print(f"Warning: Feature {feature} not found in dataframe")
                    continue

                # Get failure data for this specific failure type
                failure_data = df[df['Failure_Reason'] == failure_type][feature]

                # Skip if no data for this failure type
                if len(failure_data) == 0:
                    print(f"No data for failure type: {failure_type}")
                    continue

                # Get normal data (no failure)
                normal_data = df[df['Failure_Reason'] == 'No Failure'][feature]

                # Calculate statistics
                failure_min = failure_data.min()
                failure_max = failure_data.max()
                failure_mean = failure_data.mean()
                normal_min = normal_data.min()
                normal_max = normal_data.max()
                normal_mean = normal_data.mean()

                # Determine threshold direction based on the nature of the failure
                if 'Low' in failure_type:
                    # For "Low" failures (e.g., Low Oil Level), we expect values below normal
                    direction = "<"
                    # Find the threshold where normal values transition to failure values
                    threshold = max(normal_min, failure_max)

                    # If there's overlap, find the optimal separation point
                    if failure_max > normal_min:
                        overlap_range = np.linspace(normal_min, failure_max, 100)
                        best_separation = 0
                        best_threshold = normal_min

                        for potential_threshold in overlap_range:
                            normal_below = (normal_data < potential_threshold).mean()
                            failure_below = (failure_data < potential_threshold).mean()
                            separation = failure_below - normal_below

                            if separation > best_separation:
                                best_separation = separation
                                best_threshold = potential_threshold

                        threshold = best_threshold
                else:
                    # For "High" failures (e.g., Overheating, High Pressure), we expect values above normal
                    direction = ">"
                    # Find the threshold where normal values transition to failure values
                    threshold = min(normal_max, failure_min)

                    # If there's overlap, find the optimal separation point
                    if failure_min < normal_max:
                        overlap_range = np.linspace(failure_min, normal_max, 100)
                        best_separation = 0
                        best_threshold = normal_max

                        for potential_threshold in overlap_range:
                            normal_above = (normal_data > potential_threshold).mean()
                            failure_above = (failure_data > potential_threshold).mean()
                            separation = failure_above - normal_above

                            if separation > best_separation:
                                best_separation = separation
                                best_threshold = potential_threshold

                        threshold = best_threshold

                # Store the threshold
                thresholds[failure_type] = {
                    'feature': feature,
                    'threshold': threshold,
                    'direction': direction,
                    'failure_range': f"{failure_min:.2f} to {failure_max:.2f}",
                    'failure_mean': failure_mean
                }

            return thresholds

        # Execute the analysis
        specific_thresholds = find_specific_failure_thresholds(df)

        # Print the results in a clear format
        # print("\n=== SPECIFIC FAILURE THRESHOLDS ===")
        for failure_type, details in specific_thresholds.items():
            feature = details['feature']
            threshold = details['threshold']
            direction = details['direction']

            # if direction == '>':
            #     print(f"{failure_type}: {threshold:.2f} (When {feature} exceeds this value)")
            # else:
            #     print(f"{failure_type}: {threshold:.2f} (When {feature} falls below this value)")
            #
            # print(f"  Feature: {feature}")
            # print(f"  Failure range: {details['failure_range']}")
            # print(f"  Failure mean: {details['failure_mean']:.2f}\n")

        # Create a simplified output that can be directly used in monitoring
        monitoring_rules = {}
        for failure_type, details in specific_thresholds.items():
            feature = details['feature']
            threshold = details['threshold']
            direction = details['direction']

            monitoring_rules[failure_type] = {
                'feature': feature,
                'threshold': round(threshold, 2),
                'condition': f"{feature} {direction} {threshold:.2f}"
            }

        print("=== SIMPLIFIED MONITORING RULES ===")
        for failure_type, rule in monitoring_rules.items():
            print(f"{failure_type}: {rule['condition']}")

