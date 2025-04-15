# Database setup 
import sqlite3
import os
for db_file in ['details.db']:
    if os.path.exists(db_file):
        os.remove(db_file)
def init_db():
    conn = sqlite3.connect('reviews.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reviews 
                 (id INTEGER PRIMARY KEY, review_text TEXT, sentiment TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()

def init_details_db():
    conn = sqlite3.connect('details.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reviews 
                 (id INTEGER PRIMARY KEY, review_text TEXT, sentiment TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()

def copy_reviews_to_details():
    src_conn = sqlite3.connect('reviews.db')
    dst_conn = sqlite3.connect('details.db')

    src_cursor = src_conn.cursor()
    dst_cursor = dst_conn.cursor()

    src_cursor.execute("SELECT * FROM reviews")
    rows = src_cursor.fetchall()

    for row in rows:
        review_id = row[0]
        review_text = row[1]
        sentiment = row[2]
        timestamp = row[3]

        # If review_text is null or empty, print the record and skip it
        if not review_text:
            print(f"Skipping record with ID {review_id} due to empty or null review_text: {row}")
            src_cursor.execute("DELETE FROM reviews WHERE id = ?", (review_id,))
        else:
            # Insert valid records into details.db
            dst_cursor.execute("""
                INSERT INTO reviews (id, review_text, sentiment, timestamp) 
                VALUES (?, ?, ?, ?)
            """, (review_id, review_text, sentiment, timestamp))

    dst_conn.commit()
    src_conn.commit()  # Commit the changes to reviews.db (deleting invalid records)
    src_conn.close()
    dst_conn.close()

init_db()
init_details_db()
copy_reviews_to_details()
