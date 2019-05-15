if [ -z "$ARTTFSDIR" ]; then
    echo "Need to set ARTTFSDIR"
    exit 1
fi


echo "Run-Number for Class generation: XXXX"
read run

eval 'root -l "generateClass.C(\"'$run'\")"';


#products=("fixedProbe" "trolley" "galil" "fluxgate" "surfaceCoil" "psFeedback" "fixedProbeWf" "issues" "plungingProbeWf")
products=("fixedProbe" "trolley" "galil" "fluxgate" "surfaceCoil" "psFeedback" "fixedProbeWf" "issues")

for product in "${products[@]}";
  do
    cp $product.h $product.hh
    if [[ "$OSTYPE" == "darwin"* ]];
        then
            sed -i "" -e 's/ULong64_t/unsigned long/g' $product.hh
            sed -i "" -e 's/UInt_t/unsigned int/g'     $product.hh
            sed -i "" -e 's/Double_t/double/g' $product.hh
            sed -i "" -e 's/Float_t/float/g'   $product.hh
            sed -i "" -e 's/UShort_t/unsigned short/g' $product.hh
            sed -i "" -e 's/Char_t/char/g'     $product.hh
            sed -i "" -e 's/\]\[/*/g'          $product.hh
            g++ -Wall -fpic -dynamiclib -o lib$product.dylib $product.C `root-config --cflags --glibs`
        else 
            sed -i  's/ULong64_t/unsigned long/g' $product.hh
            sed -i  's/UInt_t/unsigned int/g'     $product.hh
            sed -i  's/Double_t/double/g' $product.hh
            sed -i  's/Float_t/float/g'   $product.hh
            sed -i  's/UShort_t/unsigned short/g' $product.hh
            sed -i  's/Char_t/char/g'     $product.hh
            sed -i  's/\]\[/*/g'          $product.hh
            g++ -c -Wall -Werror -fpic $product.C `root-config --cflags --glibs`
            gcc -shared -o lib$product.so $product.o
    fi
done

