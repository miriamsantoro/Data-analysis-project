import Classification
import NNC
import Plotting
import pandas as pd

#%%
Plotting.Plot()

#%%
print ("1 feature")
l1 = Classification.LogReg(2,3)
l2 = Classification.SVM(2,3)
l3 = Classification.DTC(2,3)
l4 = Classification.KNC(2,3)
l5 = Classification.RFC(2,3)
l6 = Classification.MLP(2,3)
l7 = Classification.ABC(2,3)
l8 = Classification.GNB(2,3)
l9 = Classification.QDA(2,3)
l10 = Classification.SGD(2,3)
l11= NNC.NNC(2,3)

df1 = pd.DataFrame(data = {"LogReg": l1, "SVM": l2, "DTC": l3, "KNC": l4, "RFC": l5,"MLP": l6,
                          "ABC": l7, "GNB": l8, "QDA": l9, "SGD": l10, "NNC":l11}, index = [".9", ".8", ".5", ".25"])
print(df1)
#%%
print ("9 features")
m1 = Classification.LogReg(2,11)
m2 = Classification.SVM(2,11)
m3 = Classification.DTC(2,11)
m4 = Classification.KNC(2,11)
m5 = Classification.RFC(2,11)
m6 = Classification.MLP(2,11)
m7 = Classification.ABC(2,11)
m8 = Classification.GNB(2,11)
m9 = Classification.QDA(2,11)
m10 = Classification.SGD(2,11)
m11= NNC.NNC(2,11)

df9 = pd.DataFrame(data = {"LogReg": m1, "SVM": m2, "DTC": m3, "KNC": m4, "RFC": m5,"MLP": m6,
                          "ABC": m7, "GNB": m8, "QDA": m9, "SGD": m10, "NNC":m11}, index = [".9", ".8", ".5", ".25"])
print(df9)
#%%
print ("16 features")
n1 = Classification.LogReg(2,18)
n2 = Classification.SVM(2,18)
n3 = Classification.DTC(2,18)
n4 = Classification.KNC(2,18)
n5 = Classification.RFC(2,18)
n6 = Classification.MLP(2,18)
n7 = Classification.ABC(2,18)
n8 = Classification.GNB(2,18)
n9 = Classification.QDA(2,18)
n10 = Classification.SGD(2,18)
n11= NNC.NNC(2,18)

df16 = pd.DataFrame(data = {"LogReg": n1, "SVM": n2, "DTC": n3, "KNC": n4, "RFC": n5,"MLP": n6,
                          "ABC": n7, "GNB": n8, "QDA": n9, "SGD": n10, "NNC":n11}, index = [".9", ".8", ".5", ".25"])

print(df16)

#%%
print ("30 features")
o1 = Classification.LogReg(2,32)
o2 = Classification.SVM(2,32)
o3 = Classification.DTC(2,32)
o4 = Classification.KNC(2,32)
o5 = Classification.RFC(2,32)
o6 = Classification.MLP(2,32)
o7 = Classification.ABC(2,32)
o8 = Classification.GNB(2,32)
o9 = Classification.QDA(2,32)
o10 = Classification.SGD(2,32)
o11= NNC.NNC(2,32)

df30 = pd.DataFrame(data = {"LogReg": o1, "SVM": o2, "DTC": o3, "KNC": o4, "RFC": o5,"MLP": o6,
                          "ABC": o7, "GNB": o8, "QDA": o9, "SGD": o10, "NNC":o11}, index = [".9", ".8", ".5", ".25"])

print(df30)

#%%
Plotting.SVCPlot(2,4)
#%%
Plotting.DTCPlot(2,32)

#%%
Plotting.Histo()

#%%
print ("10 best features")
p1 = Classification.LogReg10()
p2 = Classification.SVM10()
p3 = Classification.DTC10()
p4 = Classification.KNC10()
p5 = Classification.RFC10()
p6 = Classification.MLP10()
p7 = Classification.ABC10()
p8 = Classification.GNB10()
p9 = Classification.QDA10()
p10 = Classification.SGD10()
p11= NNC.NNC10()

df10 = pd.DataFrame(data = {"LogReg": p1, "SVM": p2, "DTC": p3, "KNC": p4, "RFC": p5,"MLP": p6,
                          "ABC": p7, "GNB": p8, "QDA": p9, "SGD": p10, "NNC": p11}, index = [".9", ".8", ".5", ".25"])

print(df10)
#%%
Plotting.Plot3B()