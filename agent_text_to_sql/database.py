from sqlalchemy import create_engine, insert, MetaData, Table

from resources import data


class Database:
    def __init__(self):
        self.engine = None
        self.engine = self.get_engine()

        self.setup()

    def get_engine(self):
        if self.engine:
            return self.engine

        self.engine = create_engine("sqlite:///:memory:")
        return self.engine

    def setup(self):
        tables = {}

        metadata = MetaData()
        for table_name, table_info in data.items():
            table = Table(table_name, metadata, *table_info["columns"])
            tables[table_name] = table
        metadata.create_all(self.engine)

        for table_name, table_info in data.items():
            table = tables[table_name]
            rows = table_info["rows"]

            for row in rows:
                stmt = insert(table).values(**row)
                with self.engine.begin() as connection:
                    cursor = connection.execute(stmt)
