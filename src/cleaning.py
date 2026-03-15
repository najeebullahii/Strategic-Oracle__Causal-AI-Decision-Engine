import pandas as pd
from sklearn.preprocessing import LabelEncoder

input_path  = r"C:\Users\Najib\Documents\Najib's Projects\Causal AI decision engine\bank-full.csv"
output_path = r"C:\Users\Najib\Documents\Najib's Projects\Causal AI decision engine\bank-full-cleaned.csv"


def clean_data():

    # load the dataset — it uses semicolons instead of commas as separators
    df = pd.read_csv(input_path, sep=';')
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # rename the target column to something more descriptive
    df.rename(columns={'y': 'outcome'}, inplace=True)

    # convert yes/no columns to 1/0 so the model can read them
    for col in ['outcome', 'default', 'housing', 'loan']:
        df[col] = df[col].map({'yes': 1, 'no': 0})

    # create the treatment column — this is the core causal question
    # 1 = contacted via cellular, 0 = everything else
    df['treatment'] = df['contact'].apply(lambda x: 1 if x == 'cellular' else 0)

    # replace 'unknown' values with the most common value in each column
    for col in ['job', 'education', 'contact', 'poutcome']:
        most_common = df[col][df[col] != 'unknown'].mode()[0]
        df[col] = df[col].replace('unknown', most_common)

    # pdays uses -1 to mean "never contacted before" which is confusing
    # capture that as a proper binary column then clean up pdays itself
    df['was_previously_contacted'] = df['pdays'].apply(lambda x: 0 if x == -1 else 1)
    df['pdays'] = df['pdays'].replace(-1, 0)

    # convert month names to numbers
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    df['month'] = df['month'].map(month_map)

    # label encode remaining text columns — DoWhy needs everything numeric
    le = LabelEncoder()
    for col in ['job', 'marital', 'education', 'contact', 'poutcome']:
        df[col] = le.fit_transform(df[col])

    print(f"Cleaning done — {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Nulls: {df.isnull().sum().sum()} | Duplicates: {df.duplicated().sum()}")

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    clean_data()