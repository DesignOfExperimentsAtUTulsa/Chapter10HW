import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


y_t = np.array([160,171,175,182,184,181,188,193,195,200])
y = np.matrix.transpose(y_t)

x_t = np.array([[1 for i in range(10)],[10,15,15,20,20,20,25,25,28,30]])
x = np.matrix.transpose(x_t)
x_vals = [x_t[1][i] for i in range(len(x_t[1]))]

xtxi = np.linalg.inv(np.dot(x_t,x))
print(xtxi,'=xtxi')
beta = np.dot(xtxi,(np.dot(x_t,y)))
print(beta ,'=beta')
beta_t = np.transpose(beta)

y_hat = np.dot(x,beta)
rs = y-y_hat

n= len(x_vals)

ss_e = np.dot(y_t,y)-np.dot(np.dot(beta_t,x_t),y)
k = len(beta)-1

ss_r =np.dot(np.dot(beta_t,x_t),y)-np.sum(y)**2/len(y)
df_e = n - k - 1

ss_t = ss_e + ss_r
df_t = n-1

ms_r = ss_r/k
ms_e = ss_e/df_e

f0 = ms_r/ms_e
ppf = stats.f.ppf(0.95,k,df_e)
p_f = stats.f.cdf(f0,k,df_e)

#Equations for B interval 10.38 Montgomery Text
p = len(beta)
sigma_hat_sqrd = ss_e/(n-p)
#print('simga_hat =',sigma_hat)
t_B = stats.t.ppf(0.975,n-p)
cjj = xtxi[1][1]
print(cjj,'=cjj')
se_B = np.sqrt((sigma_hat_sqrd)*cjj)
print(se_B ,'= standard error beta')
print(sigma_hat_sqrd,'=sigma_hat')
print(se_B,'=se_B')
print(t_B,'=t_B')
beta_upper = beta[1]+t_B*se_B
print(beta_upper)
beta_lower = beta[1]-t_B*se_B
print(beta_lower)
print(beta[1])
print('95 confidence interval for Beta_1:\n',beta_lower,"< B <", beta_upper)

print(ss_e)
print(ss_r)
print(ss_t)
print(k)
print(df_e)
print(df_t)

print(ms_r)
print(ms_e)
print(f0)
print(ppf)
print(p_f)


lin_x = [min(x_vals),max(x_vals)]
lin_y = [y_hat[0],y_hat[-1]]

plt.plot(x_vals,y,'b*')
plt.plot(lin_x,lin_y)

plt.show()

plt.plot(x_vals,rs,'b*')
plt.show()