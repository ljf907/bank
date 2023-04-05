# 导入所需的库
import numpy as np
import pandas as pd
import streamlit as st
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
#from matplotlib import pyplot as plt
#import seaborn as sns
#%matplotlib inline

# 导入数据
full_data = pd.read_csv("C:\\Users\\luoji\\Desktop\\train_ctrUa4K.csv")
#full_data = pd.read_csv("C:\\Users\\Administrator\\Desktop\\作业\\银行贷款\\train_ctrUa4K.csv")
full_data.shape

#对于数值变量：使用均值或中位数进行插补 对于分类变量：使用常见众数进行插补，这里主要使用众数进行插补空值
full_data['Gender'].fillna(full_data['Gender'].value_counts().idxmax(), inplace=True)
full_data['Married'].fillna(full_data['Married'].value_counts().idxmax(), inplace=True)
full_data['Dependents'].fillna(full_data['Dependents'].value_counts().idxmax(), inplace=True)
full_data['Self_Employed'].fillna(full_data['Self_Employed'].value_counts().idxmax(), inplace=True)
full_data["LoanAmount"].fillna(full_data["LoanAmount"].mean(skipna=True), inplace=True)
full_data['Loan_Amount_Term'].fillna(full_data['Loan_Amount_Term'].value_counts().idxmax(), inplace=True)
full_data['Credit_History'].fillna(full_data['Credit_History'].value_counts().idxmax(), inplace=True)

#对于异常值需要进行处理，这里采用对数log转化处理，消除异常值的影响，让数据回归正态分布
full_data['LoanAmount_log'] = np.log(full_data['LoanAmount'])
full_data['LoanAmount_log'].hist(bins=20)
full_data['ApplicantIncomeLog'] = np.log(full_data['ApplicantIncome'])
full_data['ApplicantIncomeLog'].hist(bins=20)

#类别特征值转化为数值
#教育
full_data['Education']=full_data['Education'].map({'Not Graduate':0,'Graduate':1})
#性别
full_data['Gender']=full_data['Gender'].map({'Male':0,'Female':1})
#结婚
full_data['Married']= full_data['Married'].map({'Yes':1,'No':0})
#工作
full_data['Self_Employed']= full_data['Self_Employed'].map({'Yes':0,'No':1})
#贷款状态
full_data['Loan_Status']=full_data['Loan_Status'].map({'Y':1,'N':0})
#信用历史
#full_data['Credit_History']=full_data['Credit_History'].map({'Y ':1,'N':0})

predictors = ['Education', 'ApplicantIncome', 'Married', 'LoanAmount','Credit_History','Loan_Amount_Term', ]
#predictors = ['Education', 'ApplicantIncome', 'Married', 'LoanAmount','Credit_History','Loan_Amount_Term']
#选择目标变量：
outcome = 'Loan_Status'

#选择预测模型：
model = LogisticRegression()

#用数据训练模型：
full_1=full_data[full_data['Loan_Status'].notnull()]
#full_1[predictors] = full_1[predictors].values
#full_1[outcome] = full_1[outcome].values
model.fit(full_1[predictors], full_1[outcome])

#用生成模型生成预测值
predicted = model.predict(full_1[predictors])

#比较预测值与实际值，得到预测准确度
accuracy = metrics.accuracy_score(predicted,full_1[outcome])
print("Accuracy :  %s" % "{0:.3%}" .format(accuracy))

# 创建 Streamlit 应用程序
st.title("贷款信息预测")
st.header("请输入您的个人信息")
education = st.selectbox("教育程度", ["大学已毕业", "大学未毕业"])
education = 1 if education == "大学已毕业" else 0
print("education:",education)
income = st.slider("申请人收入/年（￥）", 0, 80000, 40000)
amount = st.slider("贷款金额/月（￥）", 0, 800, 400)
Loan_Amount_Term=st.slider("贷款期限/天", 0, 500, 250)
married = st.selectbox("婚姻状况", ["已婚", "未婚"])
married = 1 if married == "已婚" else 0
Credit_History = st.selectbox("行用记录", ["未有失信记录", "有失信记录"])
Credit_History = 1 if Credit_History == "未有失信记录" else 0
#print("married:",married)
#print("Credit_History:",Credit_History)
X_new = [[education,income,married,amount,Credit_History,Loan_Amount_Term]]
print(X_new)
# 预测结果
result = model.predict(X_new)[0]
if result == 1:
    st.success("恭喜您，可以贷款！")
else:
    st.error("很抱歉，您无法获得贷款。")
