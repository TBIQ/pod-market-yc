def create_scatter(x, y):
    xmax = np.max(x)
    df = pd.DataFrame(data=dict(place_in_line=np.ravel(x), price_per_pod=np.ravel(y)))
    return alt.Chart(df, width=800).mark_point().encode(
        x=alt.X(
            "place_in_line:Q"
            , scale=alt.Scale(domain=(0, xmax))
        )
        , y=alt.Y(
            "price_per_pod:Q"
            , scale=alt.Scale(domain=(0, 1), clamp=True)
        )
    )

def create_lines(df): 
    xmax = np.max(df.place_in_line.values)
    selection = alt.selection_multi(fields=['hyp_params'], bind='legend')
    return alt.Chart(df, width=800).mark_line().encode(
        x=alt.X(
            "place_in_line:Q"
            , scale=alt.Scale(domain=(0, xmax))
        )
        , y=alt.Y(
            "price_per_pod:Q"
            , scale=alt.Scale(domain=(0, 1), clamp=True)
        )
        , color=alt.Color(
            "hyp_params:N"
            , legend=alt.Legend(title="Hyper Parameters", orient="bottom", direction="vertical")
        )
        , opacity=alt.condition(selection, alt.value(1), alt.value(0.025))
    ).add_selection(
        selection
    )

def get_chart_scores(df_scores): 
    dmin = np.min(
        np.union1d(df_scores.train_nmse.values, df_scores.test_nmse)
    )
    dmax = np.max(
        np.union1d(df_scores.train_nmse, df_scores.test_nmse)
    )
    yscale = alt.Scale(domain=(dmin, dmax))
    scores = alt.Chart(df_scores, title="Training vs Testing Negative-MSE (higher is better)").mark_line().transform_fold(
        fold=['test_nmse', 'train_nmse'], 
        as_=['type', 'nmse']
    ).encode(
        x="alpha:Q",
        y=alt.Y("nmse:Q", scale=yscale), 
        color='type:N'
    )
    return scores 

def get_chart_weights_boxplot(df_weights): 
    wlr_df = pd.melt(
        df_weights
        , value_vars=df_weights.columns
        , var_name="feature"
        , value_name="weight"
    )
    return alt.Chart(wlr_df).mark_boxplot(extent='min-max').encode(
        x='weight:Q'
        , y='feature:O'
    )

def run_experiment(
    X
    , y
    , stratify
    , sw 
    , feature_names: List[str] 
    , degrees: List[int]
    , alphas: List[float]
    , model_params 
    , train_p: float
    , row_height: int = 300 
    , half_width: int = 600 
    , n_fit_samples: int = 50
    , n_folds: int = 4 
): 
    """
    args: 
        degrees: values of degree to test for hyperparameter optimality 
        alphas: values of alpha to test for hyperparameter optimality 
        train_percent: values of train_percent to test for hyperparameter optimality 
    returns: 
        alt.Chart object containg all plots to be used for model validation 
    """
    # TODO: Add weighted exponential decay to sample weights for regression 
    rs = 32
    scoring = make_scorer(mean_squared_error)
    rows = []
    dfs = defaultdict(list) 
    model = Pipeline([
        ('polyfeatures', PolynomialFeatures()), 
        ('scaler', StandardScaler()), # since polynomial features have wildly different ranges, scaling is important 
        ('regressor', Ridge(**model_params)) 
    ])
    folds = list(CustomStratifiedKFold(n_folds, rs, stratify).get_folds(X))
    cols = ['degree', 'alpha', 'train_nmse', 'test_nmse']
    df_scores = None 
    df_weights = None 
    for d, a in itertools.product(degrees, alphas): 
        model = Pipeline([
            # TODO: Add in component to downscale input feature 
            ('polyfeatures', PolynomialFeatures(degree=d)), 
            ('scaler', StandardScaler()), # since polynomial features have wildly different ranges, scaling is important 
            ('regressor', Ridge(**model_params, alpha=a)) 
            # TODO: Add in component to re-scale downscaled input feature 
        ])
        scores_train = []
        scores_test = []
        weights = []
        for i, (train_idx, test_idx) in enumerate(folds): 
            X_train, y_train, sw_train = X[train_idx], y[train_idx], sw[train_idx]
            X_test, y_test, sw_test = X[test_idx], y[test_idx], sw[test_idx]
            model.fit(X_train, y_train, regressor__sample_weight=sw_train) 
            y_pred_train = model.predict(X_train) 
            y_pred_test = model.predict(X_test) 
            score_train = mean_squared_error(y_train, y_pred_train, sample_weight=sw_train)
            score_test = mean_squared_error(y_test, y_pred_test, sample_weight=sw_test)
            scores_train.append(score_train) 
            scores_test.append(score_test) 
            feature_names = model[0].get_feature_names_out(input_features=['place_in_line'])
            weights.append(model[-1].coef_)
            
        if df_scores is None: 
            df_scores = pd.DataFrame(columns=cols)
        df_scores = pd.concat([
            df_scores
            , pd.DataFrame(data=dict(
                degree=[d]
                , alpha=[a]
                , train_nmse=[np.array(scores_train).mean()]
                , test_nmse=[np.array(scores_test).mean()]
            )) 
        ]) 
        
        if df_weights is None: 
            df_weights = pd.DataFrame(columns=feature_names)
        weights = np.ravel(np.array(weights).mean(axis=0))
        df_weights = pd.concat([
            df_weights
            , pd.DataFrame(data={f: [weights[i]] for i, f in enumerate(feature_names)})
        ])
    
    return df_scores, df_weights 


# degrees = list(range(8, 12))
# alphas = np.logspace(-14, -3, num=30) # for ridge 
# alphas = np.logspace(-8, -3, num=30) # for ridge 
# train_percent = .98

# df_scores, _ = run_experiment(X, y, stratify, sw, feature_names, degrees, alphas, dict(tol=1e-14), train_percent)