#!/bin/bash
R=10

for i in {1..9}; do java -jar negative-selection/negsel2.jar -self negative-selection/syscalls/snd-cert/N$R/snd-cert.train -n $R -r $i -c -l < negative-selection/syscalls/snd-cert/N$R/snd-cert.1.test > negative-selection/syscalls/snd-cert/N$R/results/snd-cert.1.$i.txt; done
for i in {1..9}; do java -jar negative-selection/negsel2.jar -self negative-selection/syscalls/snd-cert/N$R/snd-cert.train -n $R -r $i -c -l < negative-selection/syscalls/snd-cert/N$R/snd-cert.2.test > negative-selection/syscalls/snd-cert/N$R/results/snd-cert.2.$i.txt; done
for i in {1..9}; do java -jar negative-selection/negsel2.jar -self negative-selection/syscalls/snd-cert/N$R/snd-cert.train -n $R -r $i -c -l < negative-selection/syscalls/snd-cert/N$R/snd-cert.3.test > negative-selection/syscalls/snd-cert/N$R/results/snd-cert.3.$i.txt; done
