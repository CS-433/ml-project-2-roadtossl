import pandas as pd

def write_submission(predictions, output_path):
    """
    Write predictions to a CSV file in the required submission format.

    Args:
        predictions (numpy.ndarray or list): Array or list of binary predictions (-1 or 1).
        output_path (str): Path to save the submission file.
    """
    # Create a DataFrame with the correct format
    submission = pd.DataFrame({
        "Id": range(1, len(predictions) + 1),  # Generate IDs starting from 1
        "Prediction": predictions
    })

    # Save the DataFrame to a CSV file without an index
    submission.to_csv(output_path, index=False)
    print(f"Submission file saved to: {output_path} ✔️")