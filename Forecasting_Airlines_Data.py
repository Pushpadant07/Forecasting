import numpy as np
import pandas as pd
Airlines = pd.read_csv("D:\\ExcelR Data\\Assignments\\Forecasting\\Airlines_Data.csv")
Airlines.columns
Airlines.Passengers.plot()
#month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
p = Airlines["Month"][0]
p[0:3]
Airlines['months']= 0

for i in range(96):
    p = Airlines["Month"][i]
    Airlines['months'][i]= p[0:3]

month_dummies = pd.DataFrame(pd.get_dummies(Airlines['months']))
Airlines = pd.concat([Airlines,month_dummies],axis = 1)

Airlines["t"] = np.arange(1,97)

Airlines["t_squared"] = Airlines["t"]*Airlines["t"]


Airlines["log_Passenger"] = np.log(Airlines["Passengers"])
Airlines.rename(columns={"Passengers":'Passengers'}, inplace=True)
Airlines.Passengers.plot()

Airlines.columns


Train = Airlines.head(84)
Test = Airlines.tail(12)

# to change the index value in pandas data frame 
Test = Test.set_index(np.arange(1,13))

####################### Linear ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear
# 53.199236534802715

##################### Exponential ##############################

Exp = smf.ols('log_Passenger~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp
# 46.0573611031562

#################### Quadratic ###############################

Quad = smf.ols('Passengers~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad
# 48.051888979330975

################### Additive seasonality ########################

add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea
# 132.8197848142182

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 
# 26.360817612086503

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_Passenger~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea
# 140.06320204708638

################## Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('log_Passenger~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 
# 10.519172544323617

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
#                MODEL  RMSE_Values
# 0        rmse_linear    53.199237
# 1           rmse_Exp    46.057361
# 2          rmse_Quad    48.051889
# 3       rmse_add_sea   132.819785
# 4  rmse_add_sea_quad    26.360818
# 5      rmse_Mult_sea   140.063202
# 6  rmse_Mult_add_sea    10.519173

# so rmse value of Multiplicative Additive Seasonality has the least value among the models prepared so far so this is the best or significant model for further process

