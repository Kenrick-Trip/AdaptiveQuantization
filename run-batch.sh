PS3="Enter a number: "
select modelname in resnet18 resnet19; do
  if test -z "$modelname"; then
    echo "No model selected"
  else
    echo "Selected model: $modelname"
    break
  fi
done

for cpusiz in 1 2 3; do
  for memsiz in 1024MB 512MB; do
    for batchsiz in 1 2 3; do
      for quantlv in {0..4}; do
        echo "Running $modelname with cpu: $cpusiz, memory: $memsiz, batch size: $batchsiz and quant: $quantlv"
                docker run -v quantresults:/resultsets --memory=$memsiz --rm quantizationtester $modelname $cpusiz $memsiz $batchsiz $quantlv
      done
    done
  done
done
