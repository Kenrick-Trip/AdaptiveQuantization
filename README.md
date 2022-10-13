# AdaptiveQuantization
Repo for the course CS4215 Quantitative Performance Evaluation for Computing Systems

# Getting started

Make sure to create a volume first using:
```angular2html
docker volume create quantresults
```

Once that's done, we need to build and run the containers.

Building can be done using:
```angular2html
docker build . -t quanttesting
```

Then we need to run it using:

```angular2html
docker run -v quantresults:/resultsets quanttesting
```

We will need to add the hardware specs somewhere later on.