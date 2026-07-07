import sqlite3
from config import DATABASE

def create_database():
    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            image TEXT,

            disease TEXT,

            confidence REAL,

            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)

    conn.commit()

    conn.close()


def insert_prediction(image, disease, confidence):

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""

        INSERT INTO predictions(image,disease,confidence)

        VALUES(?,?,?)

    """,(image,disease,confidence))

    conn.commit()

    conn.close()


def get_history():

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""

        SELECT * FROM predictions

        ORDER BY id DESC

    """)

    rows = cursor.fetchall()

    conn.close()

    return rows


if __name__=="__main__":

    create_database()
