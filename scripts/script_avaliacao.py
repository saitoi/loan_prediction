#!/bin/bash

total=0

for i in {1..30}; do
    echo "Execução $i:"
    tempo=$( { /usr/bin/time -f "%e" make run >/dev/null 2>&1; } 2>&1 )
    echo "Tempo: $tempo s"
    total=$(echo "$total + $tempo" | bc)
done

media=$(echo "scale=6; $total / 30" | bc)
echo "Média de tempo de execução: $media s"
