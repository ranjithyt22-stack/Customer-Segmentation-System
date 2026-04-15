import pandas as pd

def load_and_clean_data(path):

    df = pd.read_excel(path)

    # Remove missing customers
    df = df.dropna(subset=['CustomerID'])

    # Remove negative quantity or price
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    return df