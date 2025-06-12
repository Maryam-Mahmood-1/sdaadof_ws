import csv

# === Enter your CSV file path here ===
csv_file_path = "/home/maryam-mahmood/udaadbot_ws/robot_logs/write_to_read_timing_20250612_120056.csv"  # <-- Replace with your actual CSV file path

def average_second_column(csv_file_path):
    values = []

    try:
        with open(csv_file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the first row

            for row in reader:
                if len(row) >= 2:
                    try:
                        value = float(row[1])
                        values.append(value)
                    except ValueError:
                        print(f"Skipping non-numeric value: {row[1]}")

        if values:
            average = sum(values) / len(values)
            print(f"Average of second column (excluding first row): {average}")
        else:
            print("No valid data found in second column.")

    except FileNotFoundError:
        print(f"File not found: {csv_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

average_second_column(csv_file_path)
