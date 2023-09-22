import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from PIL import Image
import pandas as pd
import sys
import os
from os.path import expanduser, basename
import subprocess
import sympy
import math


def three_not_aline(pt1,pt2,pt3):
    a=pt1[0]-pt2[0]
    b=pt1[1]-pt2[1]
    c=pt1[0]-pt3[0]
    d=pt1[1]-pt3[1]
    tmp=b*c-a*d
    return tmp
    
def ellipse_equation(pt1_x,pt1_y,pt2_x,pt2_y,pt3_x,pt3_y,pt4_x,pt4_y,pt5_x,pt5_y):

    a,b,c,d,e  = sympy.symbols('a b c d e')
    f1=a*pt1_x**2+b*pt1_x*pt1_y+c*pt1_y**2+d*pt1_x+e*pt1_y+1
    f2=a*pt2_x**2+b*pt2_x*pt2_y+c*pt2_y**2+d*pt2_x+e*pt2_y+1
    f3=a*pt3_x**2+b*pt3_x*pt3_y+c*pt3_y**2+d*pt3_x+e*pt3_y+1
    f4=a*pt4_x**2+b*pt4_x*pt4_y+c*pt4_y**2+d*pt4_x+e*pt4_y+1
    f5=a*pt5_x**2+b*pt5_x*pt5_y+c*pt5_y**2+d*pt5_x+e*pt5_y+1
    z=sympy.solve([f1,f2,f3,f4,f5],[a,b,c,d,e])
    xc=float((z[b]*z[e]-2*z[c]*z[d])/(4*z[a]*z[c]-z[b]*z[b]))
    yc=float((z[b]*z[d]-2*z[a]*z[e])/(4*z[a]*z[c]-z[b]*z[b]))
    
    b_longaxis_square=float(2*(z[a]*xc*xc+z[c]*yc*yc+z[b]*xc*yc-1)/(z[a]+z[c]+math.sqrt((z[a]-z[c])*(z[a]-z[c])+z[b]*z[b])))
    a_shortaxis_square=float(2*(z[a]*xc*xc+z[c]*yc*yc+z[b]*xc*yc-1)/(z[a]+z[c]-math.sqrt((z[a]-z[c])*(z[a]-z[c])+z[b]*z[b])))
    a=float(z[a])
    b=float(z[b])
    c=float(z[c])
    d=float(z[d])
    e=float(z[e])
    theta=0.5*np.arctan(b/(a-c))
    return xc,yc,a_shortaxis_square,b_longaxis_square,a,b,c,d,e,theta



class RANSAC:
    def __init__(self, x_data, y_data, n,threshold,inline_threshold,count_ellipse):
        self.x_data = x_data
        self.y_data = y_data
        self.n = n
        self.count_ellipse=count_ellipse
        self.threshold=threshold
        self.inline_min = inline_threshold
        self.best_model = None

    def random_sampling(self):
        while True:
            sample = []
            save_ran = []
            count = 0
            while True:
                ran = np.random.randint(len(self.x_data))

                if ran not in save_ran:
                    sample.append((self.x_data[ran], self.y_data[ran]))
                    save_ran.append(ran)
                    count += 1

                    if count == 5:
                        break
        
            pt1 = sample[0]
            pt2 = sample[1]
            pt3 = sample[2]
            pt4 = sample[3]
            pt5 = sample[4]
            
            tmp=three_not_aline(pt1,pt2,pt3)
            if abs(tmp)>1:
                break
            tmp=three_not_aline(pt1,pt2,pt4)
            if abs(tmp)>1:
                break
            tmp=three_not_aline(pt1,pt2,pt5)
            if abs(tmp)>1:
                break
            tmp=three_not_aline(pt2,pt3,pt4)
            if abs(tmp)>1:
                break
            tmp=three_not_aline(pt2,pt3,pt5)
            if abs(tmp)>1:
                break
            tmp=three_not_aline(pt3,pt4,pt5)
            if abs(tmp)>1:
                break
        return sample

    def make_model(self, sample):
        pt1 = sample[0]
        pt2 = sample[1]
        pt3 = sample[2]
        pt4 = sample[3]
        pt5 = sample[4]
        pt1_x=pt1[0]
        pt1_y=pt1[1]
        pt2_x=pt2[0]
        pt2_y=pt2[1]
        pt3_x=pt3[0]
        pt3_y=pt3[1]
        pt4_x=pt4[0]
        pt4_y=pt4[1]
        pt5_x=pt5[0]
        pt5_y=pt5[1]
        xc,yc,a_shortaxis_square,b_longaxis_square,a,b,c,d,e,theta=ellipse_equation(pt1_x,pt1_y,pt2_x,pt2_y,pt3_x,pt3_y,pt4_x,pt4_y,pt5_x,pt5_y)
        isellipse=b*b-a*c
        print(pt1[0],pt1[1],pt2[0],pt2[1],pt3[0],pt3[1],pt4[0],pt4[1],pt5[0],pt5[1],xc,yc,a_shortaxis_square,b_longaxis_square,a,b,c,d,e,theta,isellipse)
        
        return xc,yc,a_shortaxis_square,b_longaxis_square,a,b,c,d,e,theta,isellipse
       
        
    def eval_model(self, model):
        inline=0
        force_length=0
        xc,yc,a_shortaxis_square,b_longaxis_square,a,b,c,d,e,theta,isellipse= model
        force_length=math.sqrt(abs(b_longaxis_square-a_shortaxis_square))
        pointright_y=force_length*math.sin(theta)+yc
        pointright_x=force_length*math.cos(theta)+xc
        pointleft_y=yc-force_length*math.sin(theta)
        pointleft_x=xc-force_length*math.cos(theta)
        if(isellipse<0 and abs(force_length)<0.1 ):
           self.count_ellipse=self.count_ellipse+1
           #print(b_longaxis_square,a_shortaxis_square)
           for i in range(len(self.x_data)):
               
               if b_longaxis_square<a_shortaxis_square:
                  dis=math.sqrt((self.x_data[i]-pointright_x)*(self.x_data[i]-pointright_x)+(self.y_data[i]-pointright_y)*(self.y_data[i]-pointright_y))+math.sqrt((self.x_data[i]-pointleft_x)*(self.x_data[i]-pointleft_x)+(self.y_data[i]-pointleft_y)*(self.y_data[i]-pointleft_y))-2*math.sqrt(a_shortaxis_square)
               else:
                  dis=math.sqrt((self.x_data[i]-pointright_x)*(self.x_data[i]-pointright_x)+(self.y_data[i]-pointright_y)*(self.y_data[i]-pointright_y))+math.sqrt((self.x_data[i]-pointleft_x)*(self.x_data[i]-pointleft_x)+(self.y_data[i]-pointleft_y)*(self.y_data[i]-pointleft_y))-2*math.sqrt(b_longaxis_square)
               if (dis<5 and dis>(-5)):
                   inline=inline+1
        else:
          inline=0
        return inline,force_length

    def execute_ransac(self):
        for i in range(self.n):
           model = self.make_model(self.random_sampling())
           inline_temp,force_length=self.eval_model(model)
           print(inline_temp)
           print('c:%d'%self.count_ellipse)
           print(force_length)
           
           if self.inline_min < inline_temp:
              self.best_model = model
              self.inline_min = inline_temp
            


def get_next_circle(a,b,r,x_data,y_data,threshold):
    dis_list=[]
    dis_nr=0
    x_data_nr=[]
    dis_list_new=[]
    x_data_nr_new=[]
    for i in range(len(x_data)):
        dis = np.sqrt((x_data[i]-a)**2 + (y_data[i]-b)**2)
        if dis not in dis_list:
           dis_nr=dis_nr+1
           dis_list.append(dis)
           x_data_nr.append(1)
        if dis in dis_list:
           dis_index=dis_list.index(dis)
           x_data_nr[dis_index]+=1
    
    proc_file.write("distance_Nr: %d\n"%dis_nr)
    dis_list_mark=[0]*dis_nr
    count_c=0
    dis_list_old=dis_list
    x_data_nr_old=x_data_nr
    x_data_nr=[]
    dis_list.sort()
    for dis in dis_list_old:
        dis_index=dis_list.index(dis)
        x_data_nr.append(x_data_nr_old[dis_index])
  
   
    for i in range(dis_nr):
       if(dis_list_mark[i]!=1):
          for j in range(dis_nr):
              if(((dis_list[j]-dis_list[i])<=threshold) and dis_list_mark[j]!=1):
                
                if(dis_list_mark[i]!=1):
                 count_c=count_c+1
                 dis_list_new.append(dis_list[i])
                 if(j!=i):
                  x_data_nr_new.append(x_data_nr[i]+x_data_nr[j])
                 else:
                  x_data_nr_new.append(x_data_nr[i])
                 dis_list_mark[i]=1
                 dis_list_mark[j]=1
                else:
                 dis_index=dis_list_new.index(dis_list[i])
                 x_data_nr_new[dis_index]+=x_data_nr[j]
                 dis_list_mark[j]=1
   
    proc_file.write("distance_Nr_1st_filter: %d\n"%count_c)
    
    dis_list_new_filter=[]
    for i in range(count_c):
        if (x_data_nr_new[i]>filter_ring_points_threshold):
           dis_list_new_filter.append(dis_list_new[i])
    proc_file.write("distance_Nr_2nd_filter: %d\n"%(len(dis_list_new_filter)))
    proc_file.write("distance_list_2nd_filter: \n")
    for i in range(len(dis_list_new_filter)):
        proc_file.write(f'{dis_list_new_filter[i]}'+'\n')
    dis_list_final=find_accurate(dis_list_new_filter)
    return dis_list_final
        
def find_accurate(dis_list):
    count=0
    dis_list_start=[]
    dis_list_end=[]
    for i in range(len(dis_list)-1):
        if (dis_list[i+1]-dis_list[i]<2):
            count=count+1
        else:
           dis_list_start.append(dis_list[i-count])
           dis_list_end.append(dis_list[i])
           count=0
    return dis_list_start,dis_list_end
 
    
def readParams():
    global StartNr,EndNr,samplename,file_path,dark_file_path,nFrames,nFrames_intg,OutFolder,OutFolder_ring,dataType,skipbytes,nFrames_dark,type_bytes,ext,NxPixel,Nypixel,px,width,Ransac_SamplingNr,filter_ring_points_threshold,Ransac_dis_threshold,ring_distance_threshold,Intg_Folder,Ransac_inline_threshold,pad,Data_type,frame_for_Ransac,doIntegration,paramFN,Radius,Eta,RMax,RMin,EtaMax,EtaMin,etaBinSize,rBinSize,rWidth,etaWidth,raw_filter
    print("Load parameters from "+filePath_param+" to this classification workflow")
    f= open(filePath_param,'r')
    paramContents=f.readlines()
   
    for line in paramContents:
        if line == '\n':
            continue
        if line[0] == '#':
            continue
        if 'startNr' == line.split()[0]:
            StartNr = int(line.split()[1])
        if 'endNr' == line.split()[0]:
            EndNr = int(line.split()[1])
        if 'pad' == line.split()[0]:
            pad = int(line.split()[1])
        if 'fStem' == line.split()[0]:
            samplename = str((line.split()[1]).split('/')[-1])
            file_path = str((line.split()[1]))
        if 'nFrames' == line.split()[0]:
            nFrames = int(line.split()[1])
        if 'nFrames_intg' == line.split()[0]:
            nFrames_intg = int(line.split()[1])
        if 'darkfn' == line.split()[0]:
            dark_file_path= str((line.split()[1]))
        if 'nFrames_dark' == line.split()[0]:
            nFrames_dark = int(line.split()[1])
        if 'OutFolder'==line.split()[0]:
            OutFolder=str(line.split()[1])
        if 'OutFolder_ring'==line.split()[0]:
            OutFolder_ring=str(line.split()[1])
        #if 'dataType' == line.split()[0]:
            #dataType = str(line.split()[1])
        if 'HeaderBytes' == line.split()[0]:
            skipbytes = int(line.split()[1])
        if 'frame_for_Ransac'== line.split()[0]:
            frame_for_Ransac = int(line.split()[1])
        if 'ext'==line.split()[0]:
            ext=str(line.split()[1])
        if 'NxPixel' == line.split()[0]:
            NxPixel = int(line.split()[1])
        if 'NyPixel' == line.split()[0]:
            Nypixel = int(line.split()[1])
        if 'px' == line.split()[0]:
            px = int(line.split()[1])
        if 'ring_width' == line.split()[0]:
            width = int(line.split()[1])
        if 'Ransac_SamplingNr' == line.split()[0]:
            Ransac_SamplingNr = int(line.split()[1])
        if 'Ransac_ring_dis_threshold' == line.split()[0]:
            Ransac_dis_threshold = int(line.split()[1])
        if 'filter_ring_points_threshold' == line.split()[0]:
            filter_ring_points_threshold = int(line.split()[1])
        if 'Ransac_inline_threshold' == line.split()[0]:
            Ransac_inline_threshold = int(line.split()[1])
        if 'ring_distance_threshold' == line.split()[0]:
            ring_distance_threshold = int(line.split()[1])
        if 'Integration_Folder'==line.split()[0]:
            Intg_Folder=str(line.split()[1])
        if 'doIntegration' == line.split()[0]:
            doIntegration = int(line.split()[1])
        if 'paramFNforIntg'==line.split()[0]:
            paramFN=str(line.split()[1])
        if 'RMin'==line.split()[0]:
            RMin=float(line.split()[1])
        if 'RMax'==line.split()[0]:
            RMax=float(line.split()[1])
        if 'EtaMin'==line.split()[0]:
            EtaMin=float(line.split()[1])
        if 'EtaMax'==line.split()[0]:
            EtaMax=float(line.split()[1])
        if 'RBinSize'==line.split()[0]:
            rBinSize=float(line.split()[1])
        if 'EtaBinSize'==line.split()[0]:
            etaBinSize=float(line.split()[1])
        if 'etaWidth'==line.split()[0]:
            etaWidth=float(line.split()[1])
        if 'Rwidth'==line.split()[0]:
            rWidth=float(line.split()[1])
        if 'rads'==line.split()[0]:
            Radius=float(line.split()[1])
        if 'etas'==line.split()[0]:
            Eta=float(line.split()[1])
        if 'raw_filter'==line.split()[0]:
            raw_filter=float(line.split()[1])
            
            
    if(ext==".raw"):
       Data_type="int32"
       type_bytes=4
    if(ext==".ge3"):
       Data_type="int16"
       type_bytes=2
    if(ext==".edf.ge3"):
       Data_type="int16"
       type_bytes=2
    if(ext==".ge5"):
       Data_type="int16"
       type_bytes=2
    
def pre_proc():
     global file_fullpath,PNG_fn
     file_fullpath=file_path+'_'+str(StartNr).zfill(pad)+ext
     print("file_fullpath: ",file_fullpath)
     raw_data=np.fromfile(file_fullpath,dtype=str(Data_type))
     print("read raw data done.")
     skip=int(skipbytes/(type_bytes))
     raw_data=raw_data[skip:]
     raw_data=np.reshape(raw_data,(nFrames,NxPixel,Nypixel))
     
     dark_data=np.fromfile(dark_file_path,dtype=str(Data_type))
     dark_data=dark_data[skip:]
     dark_data=np.reshape(dark_data,(nFrames_dark,NxPixel,Nypixel))
     dark_data=dark_data[0,:,:]
     print("read dark data done.")
     
     raw_data_now=raw_data[frame_for_Ransac,:,:]
     raw_data_now=np.nan_to_num(raw_data_now,nan=0.0)
     '''
     im = Image.fromarray(raw_data_now)
     PNG_fn=OutFolder+'/'+samplename+'_'+str(StartNr).zfill(pad)+'_nframe_'+str(frame_for_Ransac)+'raw'+".png"
     im.save(PNG_fn)
     
     im = Image.fromarray(dark_data)
     PNG_fn=OutFolder+'/'+samplename+'_'+str(StartNr).zfill(pad)+'_nframe_'+str(frame_for_Ransac)+'dark'+".png"
     im.save(PNG_fn)
     '''
     for i in range(NxPixel):
         for j in range(Nypixel):
             if(raw_data_now[i,j]<dark_data[i,j]):
                raw_data_now[i,j]=dark_data[i,j]
    
     raw_data_now=raw_data_now-dark_data
     
     raw_data_now[raw_data_now>10000]=0
     
     '''
     raw_data_now[1800:2048,1700:2048]=0
     '''
     raw_data_now_std=np.dtype(float)
     raw_data_now_mean=np.dtype(float)
     raw_data_now_std=np.std(raw_data_now)
     raw_data_now_mean=np.mean(raw_data_now)
     print("raw data std: ",raw_data_now_std," raw data mean: ",raw_data_now_mean)
     
     for i in range(NxPixel):
        for j in range(Nypixel):
            if(raw_data_now[i,j]<(raw_data_now_std+raw_data_now_mean)):
              raw_data_now[i,j]=0
            else:
                raw_data_now[i,j]=raw_data_now[i,j]-raw_data_now_std-raw_data_now_mean
     raw_data_now[raw_data_now<raw_filter]=0
     im = Image.fromarray(raw_data_now)
     PNG_fn=OutFolder+'/'+samplename+'_'+str(StartNr).zfill(pad)+'_nframe_'+str(frame_for_Ransac)+".png"
     im.save(PNG_fn)
     print("Done preprocessed data.")
     
if len(sys.argv)!=2:
    print("Usage:\ anapy ransac_circle_link_v5.py paramFN")
filePath_param=sys.argv[1]
readParams()

os.makedirs(f'{OutFolder}',exist_ok=True)
os.makedirs(f'{OutFolder}'+'/'f'{OutFolder_ring}'+'/powder/',exist_ok=True)

pre_proc()
img=Image.open(PNG_fn)
img_array=np.array(img)
x_data=[]
y_data=[]
    #img_array[0:2,:]=0
    
for i in range(NxPixel):
    for j in range(Nypixel):
      if img_array[i,j]>0:
       x_data.append(i)
       y_data.append(j)
count_ellipse=0
plt.scatter(x_data, y_data)
ransac = RANSAC(x_data, y_data, Ransac_SamplingNr,Ransac_dis_threshold,Ransac_inline_threshold,count_ellipse)
ransac.execute_ransac()
xc,yc,a_shortaxis_square,b_longaxis_square,a,b,c,d,e= ransac.best_model[0], ransac.best_model[1], ransac.best_model[2],ransac.best_model[3],ransac.best_model[4],ransac.best_model[5],ransac.best_model[6], ransac.best_model[7],ransac.best_model[8]

print(xc)
print(yc)
print(a_shortaxis_square)
print(b_longaxis_square)
print(a)
print(b)
print(c)
print(d)
print(e)
print(inline)




proc_file=open(filePath_param+'.upd','w')
#proc_file.write("%f %f %f\n"%(a,b,r))

dis_list_final=get_next_circle(a,b,r,x_data,y_data,ring_distance_threshold)
dis_list_final_start,dis_list_final_end=dis_list_final
rads=[]
for circle_nr in range(len(dis_list_final_start)):
    rads.append((dis_list_final_end[circle_nr]+dis_list_final_start[circle_nr])/2)
    circle = plt.Circle((a, b), radius=rads[circle_nr], color='r', fc='y', fill=False)
    plt.gca().add_patch(circle)
plt.axis('scaled')
plt.tight_layout()
plt.savefig(f'{samplename}'+'_'+str(StartNr).zfill(pad)+f'{ext}'+'.png')

proc_file.write("distance_Nr_final: %d\n"%len(rads))
proc_file.write("distance_list_final: \n")
for i in range(len(rads)):
     proc_file.write(f'{rads[i]}'+'\n')
proc_file.close()

rings_dis=[]
radhigh=[]
radlow=[]
for ringNr in range(len(rads)-1):
    rings_dis.append(dis_list_final_start[ringNr+1]-dis_list_final_end[ringNr])
    
radlow.append(rads[0]-width)
for ringNr in range(len(rads)-1):
    if rings_dis[ringNr]>=(2*width):
        radhigh.append(rads[ringNr]+width)
        radlow.append(rads[ringNr+1]-width)
    else:
        radhigh.append(dis_list_final_end[ringNr]+rings_dis[ringNr]/2)
        radlow.append(dis_list_final_start[ringNr+1]-rings_dis[ringNr]/2)
radhigh.append(rads[len(rads)-1]+width)

print(rads)
print(rings_dis)
print(dis_list_final_start)
print(dis_list_final_end)
print(radlow)
print(radhigh)
'''
nRBin=int((RMax-RMin)/rBinSize)
nEtaBin=int((EtaMax-EtaMin)/etaBinSize)
nFiles = EndNr - StartNr + 1
df=pd.DataFrame({})
df_powder=pd.DataFrame({})

for fileNr in range(StartNr,EndNr):
    
    if (doIntegration==1):
       updF = open(paramFN, 'a')
       updF.write(f'nFiles {nFiles}\n')
       updF.write(f'RadiusToFit {Radius} {rWidth}\n')
       updF.write(f'EtaToFit {Eta} {etaWidth}\n')
       updF.write(f'nRBin {nRBin}\n')
       updF.write(f'nEtaBin {nEtaBin}\n')
       updF.close()
       
       os.makedirs(f'{OutFolder}'+'/'+f'{Intg_Folder}',exist_ok=True)
       
       cmd1 = f'{expanduser("~/opt/MIDAS/DT/bin/DetectorMapper")} {paramFN}'
       subprocess.call(cmd1,shell=True)
       cmd = f'{expanduser("~/opt/MIDAS/DT/binmultipeak/IntegratorOMP_lineout")} {paramFN}'
       subprocess.call(cmd,shell=True)
    
    
    baseFname = samplename+'_'+str(fileNr).zfill(pad)+ext
    retafn = OutFolder+'/'+Intg_Folder+'/'+baseFname+'.REtaAreaMap.csv'
    nEtaBins = int(open(retafn).readline().split()[1])
    REtaMap = np.genfromtxt(retafn,skip_header=2)
    goodCoord=[]
    data_powder=[]
    data_sum=[]
    data_mean=[]
    
    data_powder_sum=[]
    data_powder_mean=[]
        if(fileNr%2==1):
       nFrames=10
       nFrames_intg=10
    else:
       nFrames=1
       nFrames_intg=1
    
    f = OutFolder+'/'+Intg_Folder + '/' + baseFname + '_integrated.bin'
    f = open(f,'rb')
    data_all = np.fromfile(f,dtype=np.double,count=(nRBin*nEtaBin*nFrames_intg))
    f.close()
    data_all = np.reshape(data_all,(nFrames_intg,nRBin*nEtaBin))
   
    
    for ringNr in range(0,len(rads)):
        goodCoord = np.where(np.logical_and(REtaMap[:,0]>=radlow[ringNr],REtaMap[:,0]<=radhigh[ringNr]))
       
        if (len(goodCoord[0])==0):
            continue
        nrows = int(len(goodCoord[0])/nEtaBins)
        #print(nrows)
        #for iFrame in range(1,nFrames):
        count_loop=0
        
        for iFrame in range(0,nFrames):
            
            data=data_all[iFrame,:]
            #data=np.transpose(data,(1,0))
            #print(data.shape)
            #print(goodCoord)
            data = data[goodCoord]
            data[data<1] = 1
            data = data.reshape(nrows,nEtaBin)
            #print(data)
            I = np.log(data)
            #I=data
            I8 = ((I / I.max()) * 255.9).astype(np.uint8)
            
    
            outfName = OutFolder+'/'+OutFolder_ring + '/1/' + baseFname + '_framenr_' + str(iFrame)+'_radius_'+str(ringNr) + '.jpg'
            img = Image.fromarray(I8)
            img.save(outfName)
            
            if(count_loop%10!=0):
              data_powder=data_powder+data
            else:
              data_powder=data
            if(count_loop%10==9):
              I = np.log(data_powder)
              I8 = ((I / I.max()) * 255.9).astype(np.uint8)
              outfName = OutFolder+'/'+OutFolder_ring + '/powder/' + baseFname + '_framenr_' + str(iFrame-9)+'to'+str(iFrame)+'_radius_'+str(ringNr) + '.jpg'
              img = Image.fromarray(I8)
              img.save(outfName)
            
              for row_now in range(nrows):
                   data_powder_now=data_powder[row_now,:]
                   if(row_now==0):
                      data_powder_sum=data_powder_now
                   else:
                      data_powder_sum+=data_powder_now
              data_powder_mean=data_powder_sum/nrows
              data_powder_mean=np.array([data_powder_mean])
              df_powder=df_powder.append(pd.DataFrame(data_powder_mean),ignore_index=True)
            
            for row_now in range(nrows):
                data_now=data[row_now,:]
                if(row_now==0):
                  data_sum=data_now
                else:
                  data_sum+=data_now
            data_mean=data_sum/nrows
            data_mean=np.array([data_mean])
            df=df.append(pd.DataFrame(data_mean),ignore_index=True)
            count_loop+=1
            
df.to_pickle("U3O8_600a_fn70_1_frame.pkl")
df_powder.to_pickle("U3O8_600a_fn70_10_frame.pkl")
'''
