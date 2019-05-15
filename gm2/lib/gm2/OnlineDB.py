import psycopg2
from datetime import datetime
from gm2 import DB
from gm2 import util
from gm2 import np

class OnlineDB(DB, object):
    """This class provides a set of function to query the g-2 online databank.

    Attributes:
        """
    def __init__(self, user="gm2_reader", host="g2db-priv", database="gm2_online_prod", port=5433):
        """Constructor connects to the online database.
        
        Args:
            user (str) : db user. Current default gm2_reader. 
            host(str) : db host. Default g2db-priv. 
            database(str) : db database. Default gm2_online_prod. 
            port(int) : db port. Defaults to 5433.
        """
        super(OnlineDB, self).__init__(user=user, host=host, database=database, port=port)

    def getTemperature(self, start, end):
        """Return"""
        start_ts = util.ts2datetime(np.array([start]))[0]
        end_ts  =  util.ts2datetime(np.array([end]))[0]
        #if mode in ['all']:
        #   mode_ = 'step1_voltage, step2_voltage, total_current, pos_2step_current, pos_1step_current, neg_2step_current, neg_1step_current'
        #else:
        #   mode_ = mode
        tmp_hall = {
                'NorthWall' : {'East'       :  {'channel' : 'mscb174_Temp_P1', 'id' : 4, 'times' : [], 'values' : []},
                               'MiddleEast' :  {'channel' : 'mscb174_Temp_P1', 'id' : 5, 'times' : [], 'values' : []},
                               'MiddleWest' :  {'channel' : 'mscb174_Temp_P1', 'id' : 6, 'times' : [], 'values' : []},
                               'Weast'      :  {'channel' : 'mscb174_Temp_P1', 'id' : 7, 'times' : [], 'values' : []},
                               },
                'EastWall' : {'Midle'       :  {'channel' : 'mscb174_Temp_P5', 'id' : 2, 'times' : [], 'values' : []},
                              'South'       :  {'channel' : 'mscb174_Temp_P5', 'id' : 3, 'times' : [], 'values' : []},
                              },
                'WestWall' : {'Middle'      :  {'channel' : 'mscb174_Temp_P5', 'id' : 4, 'times' : [], 'values' : []},
                              'North'       :  {'channel' : 'mscb174_Temp_P5', 'id' : 5, 'times' : [], 'values' : []},
                              'Middle2'     :  {'channel' : 'mscb174_Temp_P5', 'id' : 6, 'times' : [], 'values' : []},
                              'South'       :  {'channel' : 'mscb174_Temp_P5', 'id' : 7, 'times' : [], 'values' : []}
                              }
                }

        tmp_yoke = {
                'A' : {'Top'    : {'channel' : 'mscb323_Temp_P1',   'id' : 0, 'times' : [], 'values' : []},
                       'Bottom' : {'channel' : 'mscb323_Temp_P1',   'id' : 2, 'times' : [], 'values' : []},
                       'Back'   : {'channel' : 'mscb323_Temp_P1',   'id' : 1, 'times' : [], 'values' : []},
                       'Air'    : {'channel' : 'mscb323_Temp_P1',   'id' : 3, 'times' : [], 'values' : []},
                       'Vac'    : {'channel' : 'mscb319_PT1000_P0', 'id' : 0, 'times' : [], 'values' : []}
                    },
                'B' : {'Top'    : {'channel' : 'mscb13e_Temp_P1',   'id' : 0, 'times' : [], 'values' : []},
                       'Bottom' : {'channel' : 'mscb13e_Temp_P1',   'id' : 2, 'times' : [], 'values' : []},
                       'Back'   : {'channel' : 'mscb13e_Temp_P1',   'id' : 1, 'times' : [], 'values' : []},
                       'Air'    : {'channel' : 'mscb13e_Temp_P1',   'id' : 3, 'times' : [], 'values' : []},
                       'Vac'    : {'channel' : 'mscb319_PT1000_P0', 'id' : 1, 'times' : [], 'values' : []}
                       },
                'C' : {'Top'    : {'channel' : 'mscb323_Temp_P1',   'id' : 4, 'times' : [], 'values' : []},
                       'Bottom' : {'channel' : 'mscb323_Temp_P1',   'id' : 6, 'times' : [], 'values' : []},
                       'Back'   : {'channel' : 'mscb323_Temp_P1',   'id' : 5, 'times' : [], 'values' : []},
                       'Air'    : {'channel' : 'mscb323_Temp_P1',   'id' : 7, 'times' : [], 'values' : []},
                       'Vac'    : {'channel' : 'mscb319_PT1000_P1', 'id' : 0, 'times' : [], 'values' : []}      
                       },
                'D' : {'Top'    : {'channel' : 'mscb13e_Temp_P1',   'id' : 4, 'times' : [], 'values' : []},
                       'Bottom' : {'channel' : 'mscb13e_Temp_P1',   'id' : 6, 'times' : [], 'values' : []},
                       'Back'   : {'channel' : 'mscb13e_Temp_P1',   'id' : 5, 'times' : [], 'values' : []},
                       'Air'    : {'channel' : 'mscb13e_Temp_P1',   'id' : 7, 'times' : [], 'values' : []},
                       'Vac'    : {'channel' : 'mscb319_PT1000_P1', 'id' : 1, 'times' : [], 'values' : []}
                       },
                'E' : {'Top'    : {'channel' : 'mscb323_Temp_P2',   'id' : 0, 'times' : [], 'values' : []},
                       'Bottom' : {'channel' : 'mscb323_Temp_P2',   'id' : 2, 'times' : [], 'values' : []},
                       'Back'   : {'channel' : 'mscb323_Temp_P2',   'id' : 1, 'times' : [], 'values' : []},
                       'Air'    : {'channel' : 'mscb323_Temp_P2',   'id' : 3, 'times' : [], 'values' : []},
                       'Vac'    : {'channel' : 'mscb319_PT1000_P2', 'id' : 0, 'times' : [], 'values' : []}
                       },
                'F' : {'Top'    : {'channel' : 'mscb13e_Temp_P4',   'id' : 6, 'times' : [], 'values' : []},
                       'Bottom' : {'channel' : 'mscb13e_Temp_P2',   'id' : 0, 'times' : [], 'values' : []},
                       'Back'   : {'channel' : 'mscb13e_Temp_P4',   'id' : 1, 'times' : [], 'values' : []},
                       'Air'    : {'channel' : 'mscb13e_Temp_P4',   'id' : 3, 'times' : [], 'values' : []},
                       'Vac'    : {'channel' : 'mscb319_PT1000_P2', 'id' : 1, 'times' : [], 'values' : []}
                       },
                'G' : {'Top'    : {'channel' : 'mscb323_Temp_P2',   'id' : 4, 'times' : [], 'values' : []},
                       'Bottom' : {'channel' : 'mscb323_Temp_P2',   'id' : 6, 'times' : [], 'values' : []},
                       'Back'   : {'channel' : 'mscb323_Temp_P2',   'id' : 5, 'times' : [], 'values' : []},
                       'Air'    : {'channel' : 'mscb323_Temp_P2',   'id' : 7, 'times' : [], 'values' : []},
                       'Vac'    : {'channel' : 'mscb319_PT1000_P3', 'id' : 0, 'times' : [], 'values' : []}
                       },
                'H' : {'Top'    : {'channel' : 'mscb13e_Temp_P2',   'id' : 4, 'times' : [], 'values' : []},
                       'Bottom' : {'channel' : 'mscb13e_Temp_P2',   'id' : 6, 'times' : [], 'values' : []},
                       'Back'   : {'channel' : 'mscb13e_Temp_P2',   'id' : 5, 'times' : [], 'values' : []},
                       'Air'    : {'channel' : 'mscb13e_Temp_P2',   'id' : 7, 'times' : [], 'values' : []},
                       'Vac'    : {'channel' : 'mscb319_PT1000_P3', 'id' : 1, 'times' : [], 'values' : []}
                       },
                'I' : {'Top'    : {'channel' : 'mscb323_Temp_P3',   'id' : 0, 'times' : [], 'values' : []},
                       'Bottom' : {'channel' : 'mscb323_Temp_P3',   'id' : 2, 'times' : [], 'values' : []},
                       'Back'   : {'channel' : 'mscb323_Temp_P3',   'id' : 1, 'times' : [], 'values' : []},
                       'Air'    : {'channel' : 'mscb323_Temp_P3',   'id' : 3, 'times' : [], 'values' : []},
                       'Vac'    : {'channel' : 'mscb319_PT1000_P4', 'id' : 0, 'times' : [], 'values' : []}
                       },
                'J' : {'Top'    : {'channel' : 'mscb13e_Temp_P3',   'id' : 7, 'times' : [], 'values' : []},
                       'Bottom' : {'channel' : 'mscb13e_Temp_P3',   'id' : 1, 'times' : [], 'values' : []},
                       'Back'   : {'channel' : 'mscb13e_Temp_P3',   'id' : 2, 'times' : [], 'values' : []},
                       'Air'    : {'channel' : 'mscb13e_Temp_P3',   'id' : 3, 'times' : [], 'values' : []},
                       'Vac'    : {'channel' : 'mscb319_PT1000_P4', 'id' : 1, 'times' : [], 'values' : []}
                       },
                'K' : {'Top'    : {'channel' : 'mscb323_Temp_P3',   'id' : 4, 'times' : [], 'values' : []},
                       'Bottom' : {'channel' : 'mscb323_Temp_P3',   'id' : 6, 'times' : [], 'values' : []},
                       'Back'   : {'channel' : 'mscb323_Temp_P3',   'id' : 5, 'times' : [], 'values' : []},
                       'Air'    : {'channel' : 'mscb323_Temp_P3',   'id' : 7, 'times' : [], 'values' : []},
                       'Vac'    : {'channel' : 'mscb319_PT1000_P5', 'id' : 0, 'times' : [], 'values' : []},
                       },
                'L' : {'Top'    : {'channel' : 'mscb13e_Temp_P4',   'id' : 1, 'times' : [], 'values' : []},
                       'Bottom' : {'channel' : 'mscb13e_Temp_P3',   'id' : 5, 'times' : [], 'values' : []},
                       'Back'   : {'channel' : 'mscb13e_Temp_P4',   'id' : 2, 'times' : [], 'values' : []},
                       'Air'    : {'channel' : 'mscb13e_Temp_P4',   'id' : 4, 'times' : [], 'values' : []},
                       'Vac'    : {'channel' : 'mscb319_PT1000_P5', 'id' : 1, 'times' : [], 'values' : []}
                       },
                } 

        tmp_outside = {
                'Outside' : {'ES&H' : {'channel' : 'acnet_weather_gtemp', 'id' : 0, 'times' : [], 'values' : []},
                             'AP10' : {'channel' : 'acnet_weather_dtemp', 'id' : 0, 'times' : [], 'values' : []}
                            }
                }

        map_= dict()
        tmp = dict()
        tmp.update(tmp_yoke)
        tmp.update(tmp_hall)
        tmp.update(tmp_outside)
        for key in tmp:
            for subkey in tmp[key]:
                channel = tmp[key][subkey]['channel']
                if not channel in map_:
                    map_[channel] = dict()
                map_[channel][tmp[key][subkey]['id']] = [key, subkey]


        channels = map_.keys()
        query = """SELECT channel, value, time FROM g2sc_values WHERE time > '%s' AND time < '%s'  """ % (start_ts.strftime(self.formatstr), end_ts.strftime(self.formatstr))
        query += "AND channel in("
        for i, c in enumerate(channels):
            query += "'"+c+"'"
            if i < len(channels)-1:
                query += ", "
        query += ")"
        #print query
        print("Get temperature values from DB")
        data = self.query(query)
        print("Parse values")
        for d in data:
            if d[0] in map_:
                for i,v in enumerate(d[1]):
                    if i in map_[d[0]]:
                       tmp[map_[d[0]][i][0]][map_[d[0]][i][1]]['values'].append(v)
                       tmp[map_[d[0]][i][0]][map_[d[0]][i][1]]['times'].append(util.datetime2ts_dt(d[2]))
        return tmp

    def getQuadDate(self, year, month, day, hour, mins=0, seconds=0, delta=3600*2, mode = 'total_current'):
        """Return Quad data 'mode' in from start time for delta_s seocnds.
       
        Args:
            year (int) : start time: year.
            month (int) : start time: month.
            day (int) : start time: day.
            mins (int, optional) : start time: minutes.
            seconds (int, optional) : start time: seconds.
            delta (int, optional) : delta time in secinds. Defaults to 2h  = 3600*2.
            mode (strm optional) : on of 'step1_voltage', 'step2_voltage', 'total_current', 'pos_2step_current', 'pos_1step_current', 'neg_2step_current', 'neg_1step_current'. Defaults to 'total_current'.

        """
        d_ts = util.datetime2ts(year, month, day, hour, mins, seconds)
        return self.getQuad(d_ts, d_ts+delta*1e9, mode)

    def getQuadModes(self):
        """Returns all possible quad data"""
        return ['step1_voltage', 'step2_voltage', 'total_current', 'pos_2step_current', 'pos_1step_current', 'neg_2step_current', 'neg_1step_current']

    def getQuad(self, start, end, mode = 'total_current'):
        """ Returns Quad data 'mode' in the time between start and end. (timestamps)
        
        Args:
            start (float) : start timestamp in ns.
            end (float)   : end timestamp in ns.
            mode (str, optional) : on of 'all', 'step1_voltage', 'step2_voltage', 'total_current', 'pos_2step_current', 'pos_1step_current', 'neg_2step_current', 'neg_1step_current'. Defaults to 'total_current'.
       
        Returns:
            tuple : times and selected values
        """
        start_ts = util.ts2datetime(np.array([start]))[0]
        end_ts  =  util.ts2datetime(np.array([end]))[0]
        if mode in ['all']:
           mode_ = 'step1_voltage, step2_voltage, total_current, pos_2step_current, pos_1step_current, neg_2step_current, neg_1step_current'
        else:
           mode_ = mode
        query = """SELECT time, %s FROM gm2quad_status  WHERE time > '%s' AND time < '%s'  """ % (mode_, start_ts.strftime(self.formatstr), end_ts.strftime(self.formatstr))
        data = self.query(query)
        if mode in ['all']:
            return np.array([[util.datetime2ts_dt(d[0]), d[1], d[2], d[3], d[4], d[5], d[6], d[7]] for d in data])
        else:
            return np.array([[util.datetime2ts_dt(d[0]), d[1]] for d in data])

