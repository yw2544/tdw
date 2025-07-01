sudo apt update
sudo apt install -y parallel

sudo nohup Xorg :1 -config /etc/X11/xorg-1.conf & 

/# set out_dir in batch.sh (place batch.sh in TDW dir)
/# python /workspace/pipeline.py --output try

/# cd to TDW 
/# run batch generation 

/# paste command below: 

export DISPLAY=:1   
TOTAL=4 #totoal task           
PARA=2  #parallel task        

seq 0 $((TOTAL-1)) \
  | parallel --verbose -j $PARA bash batch.sh {}
