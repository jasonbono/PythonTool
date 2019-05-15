import psycopg2
from gm2 import DB
from datetime import datetime

class Runs(DB, object):
    """This class provides a set of function to query the run databank.


    Attributes:
        user (str) : db user. Current default scorrodi.
        host(str) : db host. Default g2field-server.
        database(str) : db database. Default nearline.
        """

    def __init__(self, user="scorrodi", host="g2field-server", database="nearline"):
        """ The constructor connects to the run database.
        
        Args:
            user (str) : db user. Current default scorrodi.
            host(str) : db host. Default g2field-server.
            database(str) : db database. Default nearline.
        """
        super(Runs, self).__init__(user=user, host=host, database=database)

    def times(self, runs):
        """Get start and stop times of a given set of run numbers.
        
        Args:
            runs iterable(int) : run numbers.

        Returns:
            list([start(datetime),stop(datetime)] : list with corresponding start and stop times.
        """
        times = []
        for run in runs:
            query = """SELECT start, stop FROM runs WHERE run = '%i'""" % run
            #print query
            response = self.query(query)
            if len(response) > 0:
                #print response
                times.append(response[0])
            else:
                times.append((None, None))
        return times

    def getRunAt(self, year, month, day, hour, mins=0, seconds=0):
        """Returns run number at a given time.
        
        Args:
           year (int) : year.
           month (int) : mont.
           day (int) : day.
           houer (int) : hour.
           mins (int, optional) : minutes. Deafults to 0.
           seconds (int, optional) : seconds. Defaults to 0.

        Returns:
           int : run number. None of no run present.
        """

        date = datetime(year, month, day, hour, mins, seconds)
        formatstr = "%Y-%m-%d %H:%M:%S"
        query = """SELECT run FROM runs WHERE """
        query += "stop  > '%s' AND start < '%s'; """ % (date.strftime(formatstr), date.strftime(formatstr))
        results = self.query(query)
        if len(results) >= 1:
            return results[0][0]
        else:
            return None

    def getRunsBetween(self, date_start, date_end = None, include=True):
        """Returns the run numbers between data_start and data_end.

        Args:
           date_start(datetime.datetime) : start date. Alternative [year, month, day [hh], [min], [ss]] can be supplied.
           date_end(datetime.datetime, optional): end date. Alternative [year, month, day [hh], [min], [ss]] can be supplied.
           include(bool) : if True the runs  including this dates are returned. Otherwise the runs excluding the dates.

        Returns:
           runs(list(int)) : run numbers.
            """
        if date_end is None:
            date_end = datetime.now()

        if type(date_start) !=  datetime:
            date_start = datetime(*date_start)

        if type(date_end) !=  datetime:
            date_end   = datetime(*date_end)

        formatstr = "%Y-%m-%d %H:%M:%S"
        query = """SELECT run FROM runs WHERE """
        if include:
            query += "stop  > '%s' AND start < '%s'; """ % (date_start.strftime(formatstr), date_end.strftime(formatstr))
        else:
            query += "start > '%s' AND stop < '%s'; """ %  (date_start.strftime(formatstr), date_end.strftime(formatstr))
        return [r[0] for r in self.query(query)]
