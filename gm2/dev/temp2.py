import gm2
from gm2 import plt, np

tmp = gm2.Temperature()


t_time = []
t_t    = []

with open("/Users/scorrodi/Downloads/7721177770818dat.txt") as file:
    lines = file.readlines()
    for line in lines[1:]:
        data = line.split(" ") 
        if not data[25] in ["*","**","***","****","******",""]:
            #print(data[2][0:4], data[2][4:6], data[2][6:8], data[2][8:10], data[2][10:12], (float(data[25]) -32) * 5/9)
            t_time.append(gm2.util.datetime2ts(int(data[2][0:4]), int(data[2][4:6]), int(data[2][6:8]), int(data[2][8:10]), int(data[2][10:12]), 0))
            t_t.append((float(data[25]) -32) * 5/9.)

t_time = np.array(t_time)
t_t    = np.array(t_t)

from matplotlib.dates import DateFormatter
formatter = DateFormatter('%m/%d\n%H:%M')
formatter = DateFormatter('%m/%d')

st2017 = (t_time >  gm2.util.date2ts(2017,10,1))&(t_time < gm2.util.date2ts(2017,10,28))  
st2018 = (t_time >  gm2.util.date2ts(2018,10,1))&(t_time < gm2.util.date2ts(2018,10,28))  

from matplotlib.backends.backend_pdf import PdfPages
with PdfPages('plots/tmp_oct_2017_2018.pdf') as pdf:
  for yoke in np.arange(ord('A'), ord('A')+12):
    ax = []
    f = plt.figure(figsize=[gm2.plotutil.figsize()[0] * 1.5, gm2.plotutil.figsize()[1] * 1.0])
    #for i, pos in enumerate(['Air','Top', 'Back', 'Bottom']):
    tmp_time_top, tmp_t_top = tmp.get(chr(yoke), 'Top')
    tmp_time_bot, tmp_t_bot = tmp.get(chr(yoke), 'Bottom')

    dt = gm2.util.date2ts(2018,10,1) - gm2.util.date2ts(2017,10,1)
    s2017_top = (tmp_time_top > gm2.util.date2ts(2017,10,12))&(tmp_time_top < gm2.util.date2ts(2017,10,28))
    s2018_top = (tmp_time_top > gm2.util.date2ts(2018,10,12))&(tmp_time_top < gm2.util.date2ts(2018,10,28))
    s2017_bot = (tmp_time_bot > gm2.util.date2ts(2017,10,12))&(tmp_time_bot < gm2.util.date2ts(2017,10,28))
    s2018_bot = (tmp_time_bot > gm2.util.date2ts(2018,10,12))&(tmp_time_bot < gm2.util.date2ts(2018,10,28))
    gm2.plotutil.plot_ts((tmp_time_top[s2017_top]+dt)*1e9, tmp_t_top[s2017_top] - tmp_t_bot[s2017_bot], markersize=2, label="2017")
    gm2.plotutil.plot_ts(tmp_time_top[s2018_top]*1e9,      tmp_t_top[s2018_top] - tmp_t_bot[s2018_bot], markersize=2, label="2018")
        
    #plt.legend(title=pos)
    #if len(ax) == 1:
    plt.title("Yoke "+chr(yoke))
    #if len(ax) == 1:
    plt.ylabel("temperature gradient [$^{\circ}$C]")
    #    if len(tmp_t[s2018])>0:
    #        mean = np.nanmean(tmp_t[s2018][tmp_t[s2018]<50.])
    #        plt.ylim([mean-0.8,mean+0.8])

    plt.xlabel("date")
    plt.legend()

    plt.gca().get_xaxis().set_visible(True)
    plt.gca().xaxis.set_major_formatter(formatter)
    gm2.despine()
    #f.savefig("plots/tmp_oct_2017_2018_yoke"+chr(yoke)+".png")
    #pdf.savefig(f)
    plt.show()
