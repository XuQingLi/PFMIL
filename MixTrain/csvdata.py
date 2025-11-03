import pandas as pd

# 读取并打印 CSV 文件内容
csv_file = '/home/gem/zsw/CAMELYON16/prompt.csv'
try:
    df = pd.read_csv(csv_file, error_bad_lines=False)
    print(df)
except pd.errors.ParserError as e:
    print(f"Error parsing the CSV file: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
