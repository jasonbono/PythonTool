#export ARTTFSDIR=~/Desktop


#point to the data


#export ARTTFSDIR=/Volumes/DataStorage/gm2/FieldData/data1/newg2/DataProduction/Offline/ArtTFSDir/v9_20_00
export ARTTFSDIR=/Users/bono/Desktop/tempdatadir



#export ARTTFSDIR=/Volumes/DataStorage/gm2/FieldData/mnt/nfs/g2field-server-2/newg2/DataProduction/Nearline/ArtTFSDir
#note that run 6327 was a trolley run


#set thisgm2
#. ~/Desktop/PythonTool/newest/gm2/thisgm2.sh
. ~/Desktop/PythonTool/gm2/thisgm2.sh
#compile
#cd /Users/bono/Desktop/PythonTool/newest/gm2/lib/
cd /Users/bono/Desktop/PythonTool/gm2/lib/
. compile.sh
#set thisgm2 again
#. ~/Desktop/PythonTool/newest/gm2/thisgm2.sh
. ~/Desktop/PythonTool/gm2/thisgm2.sh

