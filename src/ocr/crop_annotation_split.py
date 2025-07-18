import pandas as pd
#from sklearn.model_selection import train_test_split

# Path to your full annotations
#ann_path = '/Users/paarthmiglani/PycharmProjects/testingagent/data/crop_annotations.csv'

#df = pd.read_csv(ann_path)

# Split 80% train, 20% val, stratify if you want balanced classes (here, just random split)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Save splits
#train_df.to_csv('/Users/paarthmiglani/PycharmProjects/testingagent/data/crop_train.csv', index=False)
#val_df.to_csv('/Users/paarthmiglani/PycharmProjects/testingagent/data/crop_val.csv', index=False)

#print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")


#Since the data is scattered we cannot validate on this

