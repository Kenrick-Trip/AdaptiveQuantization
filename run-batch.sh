for cpusiz in 1 2 3;
do
  for memsiz in 128MB 256MB 512MB;
  do
    echo "Running models with cpu: $cpusiz and memory: $memsiz"
    docker run -v quantresults:/resultsets --memory=$memsiz --rm quantizationtester $cpusiz $memsiz
  done
done