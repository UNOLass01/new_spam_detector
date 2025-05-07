import os
import pandas as pd

def load_emails_from_folder(folder_path, label):
    data = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='latin1') as f:
                    text = f.read().strip()
                    data.append((label, text))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return data

def parse_all_email_datasets(base_dir):
    all_data = []

    # Enron Dataset
    for i in range(1, 7):
        enron_dir = os.path.join(base_dir, 'enron', f'enron{i}')
        ham_dir = os.path.join(enron_dir, 'ham')
        spam_dir = os.path.join(enron_dir, 'spam')
        all_data += load_emails_from_folder(ham_dir, 0)
        all_data += load_emails_from_folder(spam_dir, 1)

    # LingSpam Dataset (bare folder only)
    lingspam_dir = os.path.join(base_dir, 'lingspam', 'bare')
    for part in os.listdir(lingspam_dir):
        part_path = os.path.join(lingspam_dir, part)
        if os.path.isdir(part_path):
            for file in os.listdir(part_path):
                file_path = os.path.join(part_path, file)
                label = 1 if 'spmsg' in file else 0
                try:
                    with open(file_path, 'r', encoding='latin1') as f:
                        text = f.read().strip()
                        all_data.append((label, text))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    # SpamAssassin Dataset
    sa_dir = os.path.join(base_dir, 'spamassassin')
    all_data += load_emails_from_folder(os.path.join(sa_dir, 'easy_ham'), 0)
    all_data += load_emails_from_folder(os.path.join(sa_dir, 'hard_ham'), 0)
    all_data += load_emails_from_folder(os.path.join(sa_dir, 'spam'), 1)

    return all_data

if __name__ == "__main__":
    base_dir = "raw_data"  # Adjust this if your script is not in the root
    emails = parse_all_email_datasets(base_dir)

    df = pd.DataFrame(emails, columns=["label", "text"])
    df.to_csv("emails.csv", index=False)
    print(f"Saved {len(df)} emails to emails.csv")
