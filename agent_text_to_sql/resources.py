from sqlalchemy import Column, String, Integer, Float

data = {
    "receipts": {
        "columns": [
            Column("receipt_id", Integer, primary_key=True),
            Column("customer_name", String(16), primary_key=True),
            Column("price", Float),
            Column("tip", Float),
        ],
        "rows": [
            {
                "receipt_id": 1,
                "customer_name": "Alan Payne",
                "price": 12.06,
                "tip": 1.20,
            },
            {
                "receipt_id": 2,
                "customer_name": "Alex Mason",
                "price": 23.86,
                "tip": 0.24,
            },
            {
                "receipt_id": 3,
                "customer_name": "Woodrow Wilson",
                "price": 53.43,
                "tip": 5.43,
            },
            {
                "receipt_id": 4,
                "customer_name": "Margaret James",
                "price": 21.11,
                "tip": 1.00,
            },
        ],
    },
    "waiters": {
        "columns": [
            Column("receipt_id", Integer, primary_key=True),
            Column("waiter_name", String(16), primary_key=True),
        ],
        "rows": [
            {"receipt_id": 1, "waiter_name": "Corey Johnson"},
            {"receipt_id": 2, "waiter_name": "Michael Watts"},
            {"receipt_id": 3, "waiter_name": "Michael Watts"},
            {"receipt_id": 4, "waiter_name": "Margaret James"},
        ],
    },
}
