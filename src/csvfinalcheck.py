import pandas as pd

df = pd.read_csv("data/emails.csv")

print("📝 First 5 rows:")
print(df.head())

print("\n📊 Label distribution:")
print(df['label'].value_counts())

print("\n🚫 Missing values:")
print(df.isnull().sum())

print("\n📏 Email length stats:")
df['text_length'] = df['text'].apply(len)
print(df['text_length'].describe())
