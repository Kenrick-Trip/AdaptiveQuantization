for modelname in resnet18 resnet34; do
  for cpusiz in 4 5 6; do
    for memsiz in 1024MB 512MB; do
      for batchsiz in 1 2 4; do
        for quantlv in None symmetric affine histogram; do
          echo "Running $modelname with cpu: $cpusiz, memory: $memsiz, batch size: $batchsiz and quant: $quantlv"
                  docker run -v quantresults:/resultsets --memory=$memsiz --cpus=$cpusiz --rm quantizationtester $modelname $cpusiz $memsiz $batchsiz $quantlv
        done
      done
    done
  done
done
