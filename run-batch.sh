for modelname in resnet18 resnet34; do
  for cpusiz in 1 2 4; do
    for memsiz in 256MB 512MB 1024MB; do
      for batchsiz in 1 2 4; do
        for quantlv in None symmetric affine histogram; do
          echo "Running $modelname with cpu: $cpusiz, memory: $memsiz, batch size: $batchsiz and quant: $quantlv"
                  docker run -v quantresults:/resultsets --memory=$memsiz --cpus=$cpusiz --rm quantizationtester $modelname $cpusiz $memsiz $batchsiz $quantlv
        done
      done
    done
  done
done
