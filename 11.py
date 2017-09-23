import matplotlib.pyplot as plt
def b1(p1,p2):
    b=(35.-60.*p2)*p1-(20.-35.*p2)
    return b
    def b2(p1,p2):
        b=(60.*p1-35.)*p2+(20.-35.*p1)
        return b
p1=1.0
p2=0.0
p1_i=[]
p2_i=[]
b1_i=[]
b2_i=[]
balance=[35./60]
p1_i.append(p1)
p2_i.append(p2)
b1_i.append(b1(p1,p2))
b2_i.append(b2(p1,p2))
step_i=[]
step_i.append(0)
for i in range(1,100001):
# step1=(15.­b1(p1,p2))*0.00001
# step2=(20.­b2(p1,p2))*0.00001
    step1=0.001 # 与@Zeratul相同，采用固定步长0.1%
    step2=0.001 # 与@Zeratul相同，采用固定步长0.1%
    if ((35.-60.*p2) >= 0.):
        p1=p1+step1
    else:
        p1=p1-step1
    p1=min(1.0,p1)
    p1=max(0.0,p1)
    if((60.*p1-35.)>=0.):
        p2=p2+step2
    else:
        p2=p2-step2
    p2=min(1.0,p2)
    p2=max(0.0,p2)
    p1_i.append(p1)
    p2_i.append(p2)
    b1_i.append(b1(p1,p2))
    b2_i.append(b2(p1,p2))
    step_i.append(i)
    balance.append(35./60)
plt.plot(step_i,p1_i,"r")
plt.plot(step_i,p2_i,"g")
plt.plot(step_i,balance,"b")