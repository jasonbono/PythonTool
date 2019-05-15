import psycopg2
from datetime import datetime

class DB:
    """Baseclass which provides functionality to query the a postgres databanks.

    Attributes:
        user (str) : db user.
        host(str) : db host.
        port(int) : db port.
        database(str) : db database.
        """

    def __init__(self, user, host, database, port = None):
        self.user = user
        self.host = host
        self.port = port
        self.database = database
        self.connect()
        self.formatstr = "%Y-%m-%d %H:%M:%S"

    def __del__(self):
        self.close()

    def connect(self):
        """ Connect to the database. So far no error handling."""
        if self.port is None:
            self.conn = psycopg2.connect("dbname='%s' user='%s' host='%s' " % (self.database, self.user, self.host))
        else:
            self.conn = psycopg2.connect("dbname='%s' user='%s' host='%s' port='%i'" % (self.database, self.user, self.host, self.port))
        self.cur  = self.conn.cursor()

    def close(self):
        """Close the connection to the database"""
        self.conn.close()

    def query(self, query):
        """Query database.
        
        Args:
            query(str) : db query string. For example 'Select run FROM runs'."

        Returns:
            runs (list): the row wise response to query.
            start (list)
        """
        self.cur.execute(query)
        rows = self.cur.fetchall()
        return rows
