rm -r ./out
mkdir ./out

#for i in 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170
#for i in 60 70 80 90 100 110 120
for i in 70 90 110
do
  echo "Setup $i"
  sleep 5
  for j in {1..3}
  do
    timeout 30 ./monitor_devs.sh > "./out/$i-$j.log"
    sleep 2
  done
  echo "End $i"
done


