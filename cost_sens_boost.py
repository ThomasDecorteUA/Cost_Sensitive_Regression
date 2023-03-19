#Rescale function 
def rescale(df):
    standard_deviation_of_float_columns = []
    df_rescaled = df.copy()
    dataTypeSeries = df[features].dtypes
    float_columns = dataTypeSeries[dataTypeSeries==float].index
    for column in df.columns:
        if column in float_columns:
            if column == "const":
                standard_deviation_of_float_columns.append(1)
            else:
                standard_deviation_of_float_columns.append(df[column].std())
                df_rescaled[column] = df[column]/df[column].std()
        else:
            standard_deviation_of_float_columns.append(1)
    return df_rescaled,standard_deviation_of_float_columns

#Function to calculate convergence
def cost_function(residual,typecost,a,b):
        
    if typecost == 'linlin':
        cost = np.where(residual >= 0, a*(np.abs(residual)), b*(np.abs(residual)))
    if typecost == 'quadquad':
        cost = np.where(residual >= 0, a*(residual**2), b*(residual**2))
    if typecost == 'lin_ex':
        cost_full_function = b*(np.exp(a*residual)-a*residual-1)
        cost = np.where(cost_full_function>1e+6,1e+6,cost_full_function)
        
    mean_cost = np.mean(cost)
    return mean_cost

#IRLS function
def IRLS(typecost, X , initial_residuals , a , b, **kwargs):
    
    #Set initial arguments + initial printout + initial cost calculation
    np.seterr(all='ignore')
    scale = kwargs.get('scale', True)
    n_iter = kwargs.get('n_iter', 1000)
    epsilon = kwargs.get('epsilon', 0.00001)
        
    cost_initial = cost_function(initial_residuals,typecost,a,b)
    
    #Add in intercept column + get dimensions of X matrix
    X = statsmodels.tools.tools.add_constant(X, prepend=True, has_constant='skip')
    n = X.shape[0]
    z = X.shape[1]
 
    #Rescaling + saving the standard deviations for rescaling of parameters later on
    if scale == True:
        X_save = X.copy()
        X,X_std_scores = rescale(X)
        X_std_scores = np.array(X_std_scores)
    
    #Initialize beta vector (always start from 0 so from predictions of original model)
    Beta = np.empty([z,])
    
    #If necessary reform Y to numpy array with correct shape
    if type(initial_residuals) is not np.ndarray:
          initial_residuals = initial_residuals.to_numpy()
    initial_residuals = initial_residuals.reshape(n, 1)

    #Start of the iteration
    for i in range(0, n_iter):
        #In the first iteration we use the residuals of the original model
        #After that we obtain residuals from: initial_residuals - predictions
        if i == 0:
            residuals = initial_residuals.copy()
            s_residuals = initial_residuals.std()
            pred = np.zeros(n)
            pred = pred.reshape(n, 1)
        else:
            pred_list = X @ Beta
            pred = np.array(pred_list)
            pred.shape = [n ,1]
            
        residuals = pred - initial_residuals      
        residuals_rescaled = residuals/1
        
        #Apply derivative cost function to the residuals to get the weights
        if typecost == 'linlin':
            residuals_rescaled[residuals_rescaled==0] = 0.0000000000001
            weights = np.where(residuals_rescaled >= 0, a/(np.abs(residuals_rescaled)), b/(np.abs(residuals_rescaled)))
        if typecost == 'quadquad':
            weights = np.where(residuals_rescaled >= 0, a*2, b*2)
        if typecost == 'lin_ex':
            residuals_rescaled[residuals_rescaled==0] = 0.0000000000001
            weights_full_function = (b*a*(np.exp(a*residuals_rescaled)-1))/residuals_rescaled
            cost_full_function = b*(np.exp(a*residuals_rescaled)-a*residuals_rescaled-1)
            cost = np.where(cost_full_function>1e+6,1e+6,cost_full_function)
            weights = np.where(cost==1e+6,0,weights_full_function)
                
        #We calculate the sqrt of the weigths in order to transform the X and Y matrices
        sqrt_wi = (np.sqrt(weights))
        sqrtW_X = sqrt_wi*X
        sqrtW_initial_residuals = sqrt_wi*(initial_residuals)
        
        #We apply linear regression to the sqrtW_X and sqrtW_initial_residuals
        lr = LinearRegression(fit_intercept = False)
        lr.fit(sqrtW_X, sqrtW_initial_residuals)
        Beta[:]  = lr.coef_
        
        #Convergence 
        if i == 0:
            previous_convergence_costs = cost_function(initial_residuals,typecost,a,b)
            
        pred_list = X @ Beta
        pred = np.array(pred_list)
        pred.shape = [n ,1]
        residuals_convergence = pred - initial_residuals   
        costs_convergence = cost_function(residuals_convergence,typecost,a,b)
        
        if (previous_convergence_costs - costs_convergence) > (s_residuals*epsilon):
            if previous_convergence_costs - costs_convergence>0:
                Best_Beta = Beta.copy()
            previous_convergence_costs = costs_convergence
        else:
            if i ==0:
                Best_Beta = np.zeros([z,]) 
            else:
                if previous_convergence_costs - costs_convergence>0:
                    Best_Beta = Beta.copy()
            break
            
    if scale == True:
        Best_Beta = Best_Beta/X_std_scores
        X = X_save
    
    #Calculate the final cost of the post hoc model
    pred_list = X @ Best_Beta
    predictions = np.array(pred_list)
    predictions.shape = [n ,1]
    final_residuals_post_hoc_model = predictions - initial_residuals
    mean_cost_post_hoc = cost_function(final_residuals_post_hoc_model,typecost,a,b)
    
    return Best_Beta,mean_cost_post_hoc,cost_initial
