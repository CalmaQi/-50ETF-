import requests
import numpy
import time
while(1):
  time.sleep(1)
  time1=time.localtime( time.time() )
  mytime=str(time1[0])+'-'+str(time1[1])+'-'+str(time1[2])+' '
  mytime+=str(time1[3])+':'+str(time1[4])+':'+str(time1[5])
  print(mytime)


