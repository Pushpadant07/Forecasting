import numpy as np
import pandas as pd
Plastic = pd.read_csv("D:\\ExcelR Data\\Assignments\\Forecasting\\PlasticSales.csv")
Plastic.columns
Plastic.Sales.plot()
#month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
p = Plastic["Month"][0]
p[0:3]
Plastic['months']= 0

for i in range(60):
    p = Plastic["Month"][i]
    Plastic['months'][i]= p[0:3]

month_dummies = pd.DataFrame(pd.get_dummies(Plastic['months']))
Plastic = pd.concat([Plastic,month_dummies],axis = 1)

Plastic["t"] = np.arange(1,61)

Plastic["t_squared"] = Plastic["t"]*Plastic["t"]


Plastic["log_Sales"] = np.log(Plastic["Sales"])
Plastic.rename(columns={"Sales":'Sales'}, inplace=True)
Plastic.Sales.plot()

Plastic.columns


Train = Plastic.head(48)
Test = Plastic.tail(12)

# to change the index value in pandas data frame 
Test = Test.set_index(np.arange(1,13))

####################### Linear ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear
# 260.93781425111206

##################### Exponential ##############################

Exp = smf.ols('log_Sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp
# 268.69383850025605

#################### Quadratic ###############################

Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad
# 297.4067097272056

################### Additive seasonality ########################

add_sea = smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea
# 235.60267356646509

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Sales~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 
# 218.19387584898945

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea
# 239.6543214312083

################## Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 
# 160.68332947192974

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

#                 MODEL  RMSE_Values
# 0        rmse_linear   260.937814
# 1           rmse_Exp   268.693839
# 2          rmse_Quad   297.406710
# 3       rmse_add_sea   235.602674
# 4  rmse_add_sea_quad   218.193876
# 5      rmse_Mult_sea   239.654321
# 6  rmse_Mult_add_sea   160.683329

# so rmse value of Multiplicative Additive Seasonality has the least value among the models prepared so far so this is the best or significant model for further process
