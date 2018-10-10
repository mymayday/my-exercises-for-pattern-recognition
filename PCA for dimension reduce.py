import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def pca(im,k):                                   #im为输入的原始图像，k为选取主成分的个数（自选）
    
    x=np.array(Image.open(im).convert('L'))      #将图像转成灰度图像
    print(x.shape)
    m=np.mean(x,axis=0)                          #算出图像中每一列的均值
    x=x-m                                        #将数据中心化
    r,c=x.shape                                  #得到图像矩阵的行列数
    C=(x.dot(x.T))/r                             #计算样本协方差矩阵，这里C是一个对称矩阵
    e,e_v=np.linalg.eig(C)                       #计算协方差矩阵的特征值和特征向量，e为特征值，e_v为对应的特征向量
    e_sort= np.argsort(-e)                       #将特征值按降序排列（从大到小）
    e_v_sort=e_v[:,e_sort]                       #特征值重新排序后的特征向量
    
    if k>r:
        print('please let k smaller than the pictures row number ')
        return
    else:
        P=e_v_sort[:,:k]                        #即为所求PCA矩阵
    
    new_x=P.T.dot(x)
    recon_x=P.dot(new_x)+m
    
    return P,recon_x
    
P1,recon_x1=pca('bigben.jpg',100)
plt.imshow(recon_x1)
