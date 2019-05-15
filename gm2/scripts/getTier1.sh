kinit scorrodi@FNAL.GOV
for i in $(seq $1 $2);
  do
    rsync g2field-server:/data1/newg2/DataProduction/Nearline/ArtTFSDir/FieldGraphOut0$i\_tier1.root $ARTTFSDIR -vaP
done
