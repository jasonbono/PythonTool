BASEDIR=$(dirname "$BASH_SOURCE")
PWD=`pwd`
if [[ $BASEDIR == $PWD* ]]; then
    export GM2=$BASEDIR;
    export PYTHONPATH=$PYTHONPATH:$BASEDIR/lib/;
else  
    export GM2=$PWD/$BASEDIR;
    export PYTHONPATH=$PYTHONPATH:$PWD/$BASEDIR/lib/;
  fi
