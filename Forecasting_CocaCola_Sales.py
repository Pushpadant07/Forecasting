import numpy as np
import pandas as pd
CocaCola = pd.read_csv("D:\\ExcelR Data\\Assignments\\Forecasting\\CocaCola_Sales_Rawdata.csv")
CocaCola.columns
CocaCola.Sales.plot()
#month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
p = CocaCola["Quarter"][0]
p[0:3]
CocaCola['quarter']= 0

for i in range(42):
    p = CocaCola["Quarter"][i]
    CocaCola['quarter'][i]= p[0:3]

month_dummies = pd.DataFrame(pd.get_dummies(CocaCola['quarter']))
CocaCola = pd.concat([CocaCola,month_dummies],axis = 1)

CocaCola["t"] = np.arange(1,43)

CocaCola["t_squared"] = CocaCola["t"]*CocaCola["t"]


CocaCola["log_Sales"] = np.log(CocaCola["Sales"])
CocaCola.rename(columns={"Sales":'Sales'}, inplace=True)
CocaCola.Sales.plot()

CocaCola.columns


Train = CocaCola.head(30)
Test = CocaCola.tail(12)

# to change the index value in pandas data frame 
Test = Test.set_index(np.arange(1,13))

####################### Linear ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear
# 714.0144483818336

##################### Exponential ##############################

Exp = smf.ols('log_Sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp
# 552.2821039688192

#################### Quadratic ###############################

Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad
# 646.2715428655387

################### Additive seasonality ########################

add_sea = smf.ols('Sales~Q1_+Q2_+Q3_+Q4_',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1_','Q2_','Q3_','Q4_']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea
# 1778.0065467724

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Sales~t+t_squared+Q1_+Q2_+Q3_+Q4_',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1_','Q2_','Q3_','Q4_','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 
# 586.0533068427092

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_Sales~Q1_+Q2_+Q3_+Q4_',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea
# 1828.9238911891812

################## Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_Sales~t+Q1_+Q2_+Q3_+Q4_',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 
# 410.2497060538084

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

#               MODEL  RMSE_Values
# 0        rmse_linear   714.014448
# 1           rmse_Exp   552.282104
# 2          rmse_Quad   646.271543
# 3       rmse_add_sea  1778.006547
# 4  rmse_add_sea_quad   586.053307
# 5      rmse_Mult_sea  1828.923891
# 6  rmse_Mult_add_sea   410.249706

# so rmse value of Multiplicative Additive Seasonality has the least value among the models prepared so far so this is the best or significant model for further process