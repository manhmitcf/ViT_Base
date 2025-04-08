import os
import pandas as pd

def split_meta_data(input_csv, output_dir="data", train_csv="train.csv", val_csv="val.csv", test_csv="test.csv"):
    df = pd.read_csv(input_csv)
    df.head()
    train = df[df['Type'] == 'Train'].drop(columns=['Type'])
    val = df[df['Type'] == 'Validation'].drop(columns=['Type'])
    test = df[df['Type'] == 'Test'].drop(columns=['Type'])

    train.to_csv(os.path.join(output_dir, train_csv), index=False)
    val.to_csv(os.path.join(output_dir, val_csv), index=False)
    test.to_csv(os.path.join(output_dir, test_csv), index=False)
    print("Meta data split completed:")
    print(f"Train: {len(train)} samples, Val: {len(val)} samples, Test: {len(test)} samples")
if __name__ == "__main__":
    split_meta_data("data/meta_data.csv")