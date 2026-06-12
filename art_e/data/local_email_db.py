import logging
import os
import sqlite3
from datetime import datetime

from datasets import Dataset, Features, Sequence, Value, load_dataset
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "..", "..", "data", "enron_emails.db")

DEFAULT_REPO_ID = "corbt/enron-emails"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

SQL_CREATE_TABLES = """
DROP TABLE IF EXISTS recipients;
DROP TABLE IF EXISTS emails_fts;
DROP TABLE IF EXISTS emails;

CREATE TABLE emails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT UNIQUE,
    subject TEXT,
    from_address TEXT,
    date TEXT,
    body TEXT,
    file_name TEXT
);

CREATE TABLE recipients (
    email_id INTEGER,
    recipient_address TEXT,
    recipient_type TEXT,
    FOREIGN KEY(email_id) REFERENCES emails(id) ON DELETE CASCADE
);
"""

SQL_CREATE_INDEXES_TRIGGERS = """
CREATE INDEX idx_emails_from ON emails(from_address);
CREATE INDEX idx_emails_date ON emails(date);
CREATE INDEX idx_emails_message_id ON emails(message_id);
CREATE INDEX idx_recipients_address ON recipients(recipient_address);
CREATE INDEX idx_recipients_type ON recipients(recipient_type);
CREATE INDEX idx_recipients_email_id ON recipients(email_id);
CREATE INDEX idx_recipients_address_email ON recipients(recipient_address, email_id);

CREATE VIRTUAL TABLE emails_fts USING fts5(
    subject,
    body,
    content='emails',
    content_rowid='id'
);

CREATE TRIGGER emails_ai AFTER INSERT ON emails BEGIN
    INSERT INTO emails_fts (rowid, subject, body)
    VALUES (new.id, new.subject, new.body);
END;

CREATE TRIGGER emails_ad AFTER DELETE ON emails BEGIN
    DELETE FROM emails_fts WHERE rowid=old.id;
END;

CREATE TRIGGER emails_au AFTER UPDATE ON emails BEGIN
    UPDATE emails_fts SET subject=new.subject, body=new.body WHERE rowid=old.id;
END;

INSERT INTO emails_fts (rowid, subject, body) SELECT id, subject, body FROM emails;
"""


def download_dataset(repo_id: str) -> Dataset:
    logging.info("Attempting to download dataset from Hugging Face Hub: %s", repo_id)
    expected_features = Features(
        {
            "message_id": Value("string"),
            "subject": Value("string"),
            "from": Value("string"),
            "to": Sequence(Value("string")),
            "cc": Sequence(Value("string")),
            "bcc": Sequence(Value("string")),
            "date": Value("timestamp[us]"),
            "body": Value("string"),
            "file_name": Value("string"),
        }
    )
    dataset_obj = load_dataset(repo_id, features=expected_features, split="train")
    if not isinstance(dataset_obj, Dataset):
        raise TypeError(f"Expected Dataset, got {type(dataset_obj)}")
    logging.info("Successfully loaded dataset '%s' with %s records.", repo_id, len(dataset_obj))
    return dataset_obj


def create_database(db_path: str) -> None:
    logging.info("Creating SQLite database and tables at: %s", db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_TABLES)
    conn.commit()
    conn.close()
    logging.info("Database tables created successfully.")


def populate_database(db_path: str, dataset: Dataset) -> None:
    logging.info("Populating database %s...", db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = MEMORY;")

    record_count = 0
    skipped_count = 0
    duplicate_count = 0
    processed_emails = set()

    conn.execute("BEGIN TRANSACTION;")

    for email_data in tqdm(dataset, desc="Inserting emails"):
        assert isinstance(email_data, dict)
        message_id = email_data["message_id"]
        subject = email_data["subject"]
        from_address = email_data["from"]
        date_obj: datetime = email_data["date"]
        body = email_data["body"]
        file_name = email_data["file_name"]
        to_list_raw = email_data["to"]
        cc_list_raw = email_data["cc"]
        bcc_list_raw = email_data["bcc"]

        date_str = date_obj.strftime("%Y-%m-%d %H:%M:%S")
        to_list = [str(addr) for addr in to_list_raw if addr]
        cc_list = [str(addr) for addr in cc_list_raw if addr]
        bcc_list = [str(addr) for addr in bcc_list_raw if addr]

        if len(body) > 5000:
            logging.debug("Skipping email %s: body length > 5000 characters.", message_id)
            skipped_count += 1
            continue

        total_recipients = len(to_list) + len(cc_list) + len(bcc_list)
        if total_recipients > 30:
            logging.debug(
                "Skipping email %s: total recipients (%s) > 30.",
                message_id,
                total_recipients,
            )
            skipped_count += 1
            continue

        email_key = (subject, body, from_address)
        if email_key in processed_emails:
            logging.debug(
                "Skipping duplicate email (subject: %s..., from: %s)",
                subject[:50],
                from_address,
            )
            duplicate_count += 1
            continue
        processed_emails.add(email_key)

        cursor.execute(
            """
            INSERT INTO emails (message_id, subject, from_address, date, body, file_name)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (message_id, subject, from_address, date_str, body, file_name),
        )
        email_pk_id = cursor.lastrowid

        recipient_data = []
        for addr in to_list:
            recipient_data.append((email_pk_id, addr, "to"))
        for addr in cc_list:
            recipient_data.append((email_pk_id, addr, "cc"))
        for addr in bcc_list:
            recipient_data.append((email_pk_id, addr, "bcc"))

        if recipient_data:
            cursor.executemany(
                """
                INSERT INTO recipients (email_id, recipient_address, recipient_type)
                VALUES (?, ?, ?)
                """,
                recipient_data,
            )
        record_count += 1

    conn.commit()
    conn.close()
    logging.info("Successfully inserted %s email records.", record_count)
    if skipped_count > 0:
        logging.info("Skipped %s email records due to length or recipient limits.", skipped_count)
    if duplicate_count > 0:
        logging.info(
            "Skipped %s duplicate email records based on subject/body/from.",
            duplicate_count,
        )


def create_indexes_and_triggers(db_path: str) -> None:
    logging.info("Creating indexes and triggers for database: %s...", db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(SQL_CREATE_INDEXES_TRIGGERS)
    conn.commit()
    conn.close()
    logging.info("Indexes and triggers created successfully.")


def generate_database(overwrite: bool = False) -> None:
    logging.info(
        "Starting database generation for repo '%s' at '%s'",
        DEFAULT_REPO_ID,
        DEFAULT_DB_PATH,
    )
    logging.info("Overwrite existing database: %s", overwrite)

    db_dir = os.path.dirname(DEFAULT_DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    if overwrite and os.path.exists(DEFAULT_DB_PATH):
        logging.warning("Removing existing database file: %s", DEFAULT_DB_PATH)
        os.remove(DEFAULT_DB_PATH)
    elif not overwrite and os.path.exists(DEFAULT_DB_PATH):
        logging.warning(
            "Database file %s exists and overwrite is False. Assuming file is already generated.",
            DEFAULT_DB_PATH,
        )
        return

    dataset = download_dataset(DEFAULT_REPO_ID)
    create_database(DEFAULT_DB_PATH)
    populate_database(DEFAULT_DB_PATH, dataset)
    create_indexes_and_triggers(DEFAULT_DB_PATH)

    logging.info("Database generation process completed for %s.", DEFAULT_DB_PATH)


if __name__ == "__main__":
    generate_database(overwrite=True)
