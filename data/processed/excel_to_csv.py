import argparse
import pandas as pd

def excel_to_csv(excel_file, csv_file, sheet_name=0):
    """
    Convert a sheet from an Excel file to a CSV file.

    excel_file: path to .xlsx or .xls file
    csv_file: path to output .csv file
    sheet_name: sheet index (0 = first sheet) or sheet name as string
    """
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    df.to_csv(csv_file, index=False)
    print(f"Saved sheet {sheet_name!r} from {excel_file} to {csv_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert an Excel sheet to CSV.")
    parser.add_argument("excel_file", help="Path to the input Excel file (.xlsx / .xls)")
    parser.add_argument("csv_file", help="Path to the output CSV file")
    parser.add_argument(
        "--sheet",
        help="Sheet name or index (default: 0 = first sheet)",
        default="0",
    )

    args = parser.parse_args()

    # Try to interpret sheet as an integer index if possible
    try:
        sheet = int(args.sheet)
    except ValueError:
        sheet = args.sheet  # treat as sheet name

    excel_to_csv(args.excel_file, args.csv_file, sheet_name=sheet)

if __name__ == "__main__":
    main()
