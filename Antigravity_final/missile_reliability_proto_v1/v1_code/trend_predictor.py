import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

class TrendPredictor:
    def __init__(self, df):
        self.df = df
        self.models_population = {}
        self.models_matching = {}
        self.model_metrics = {} # Store RMSE here
        self.matched_pairs = None

    # --- Method A: Population Trend ---
    def fit_population_models(self, target_col):
        # Filter NaNs
        valid_mask = self.df[target_col].notna() & self.df['운용월'].notna()
        if valid_mask.sum() < 2:
            return # Not enough data
            
        X = self.df.loc[valid_mask, ['운용월']].values
        y = self.df.loc[valid_mask, target_col].values
        
        metrics = {}
        
        # 1. Linear Regression (Baseline)
        lin_model = LinearRegression()
        lin_model.fit(X, y)
        y_pred = lin_model.predict(X)
        metrics['Linear'] = np.sqrt(mean_squared_error(y, y_pred))
        
        # 2. Polynomial Regression (Degree 2)
        poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        poly_model.fit(X, y)
        y_pred = poly_model.predict(X)
        metrics['Polynomial'] = np.sqrt(mean_squared_error(y, y_pred))
        
        # 3. Bayesian Ridge (Probabilistic Linear)
        bayes_model = make_pipeline(PolynomialFeatures(2), BayesianRidge())
        bayes_model.fit(X, y)
        y_pred = bayes_model.predict(X)
        metrics['Bayesian'] = np.sqrt(mean_squared_error(y, y_pred))
        
        # 4. Gaussian Process Regressor (Non-parametric, Probabilistic)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=0.1, normalize_y=True)
        gpr_model.fit(X, y)
        y_pred = gpr_model.predict(X)
        metrics['GaussianProcess'] = np.sqrt(mean_squared_error(y, y_pred))
        
        # 5. Support Vector Regression (SVR)
        # Scale X and y for SVR/MLP
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_x.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        svr_model.fit(X_scaled, y_scaled)
        y_pred_scaled = svr_model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        metrics['SVR'] = np.sqrt(mean_squared_error(y, y_pred))
        
        # 6. Neural Network (MLP Regressor)
        mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        mlp_model.fit(X_scaled, y_scaled)
        y_pred_scaled = mlp_model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        metrics['NeuralNetwork'] = np.sqrt(mean_squared_error(y, y_pred))
        
        self.models_population[target_col] = {
            'Linear': lin_model,
            'Polynomial': poly_model,
            'Bayesian': bayes_model,
            'GaussianProcess': gpr_model,
            'SVR': (svr_model, scaler_x, scaler_y),
            'NeuralNetwork': (mlp_model, scaler_x, scaler_y)
        }
        self.model_metrics[target_col] = metrics

    def predict_population(self, target_col, future_months):
        if target_col not in self.models_population:
            self.fit_population_models(target_col)
            
        X_pred = np.array(future_months).reshape(-1, 1)
        results = {}
        
        # Calculate residual std for simple CI models
        valid_mask = self.df[target_col].notna() & self.df['운용월'].notna()
        if valid_mask.sum() > 0:
            X_valid = self.df.loc[valid_mask, ['운용월']].values
            y_valid = self.df.loc[valid_mask, target_col].values
        else:
            X_valid, y_valid = np.array([]), np.array([])
        
        for name, model in self.models_population[target_col].items():
            if name == 'GaussianProcess':
                y_pred, y_std = model.predict(X_pred, return_std=True)
                results[name] = (y_pred, y_pred - 1.645*y_std, y_pred + 1.645*y_std)
                
            elif name == 'Bayesian':
                y_pred, y_std = model.predict(X_pred, return_std=True)
                results[name] = (y_pred, y_pred - 1.645*y_std, y_pred + 1.645*y_std)
                
            elif name in ['SVR', 'NeuralNetwork']:
                # Unpack model and scalers
                regressor, sx, sy = model
                X_pred_scaled = sx.transform(X_pred)
                y_pred_scaled = regressor.predict(X_pred_scaled)
                y_pred = sy.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                
                # Calculate residual std on training data
                if len(X_valid) > 0:
                    X_valid_scaled = sx.transform(X_valid)
                    y_valid_pred_scaled = regressor.predict(X_valid_scaled)
                    y_valid_pred = sy.inverse_transform(y_valid_pred_scaled.reshape(-1, 1)).ravel()
                    std = np.std(y_valid - y_valid_pred)
                else:
                    std = 0
                results[name] = (y_pred, y_pred - 1.645*std, y_pred + 1.645*std)
                
            else: # Linear, Polynomial
                y_pred = model.predict(X_pred)
                # Simple CI using residual std
                if len(X_valid) > 0:
                    y_hat = model.predict(X_valid)
                    std = np.std(y_valid - y_hat)
                else:
                    std = 0
                results[name] = (y_pred, y_pred - 1.645*std, y_pred + 1.645*std)
                
        return results

    # --- Method B: Individual Matching ---
    def perform_matching(self):
        """Matches ASRP items to QIM items using PCA distance."""
        if self.matched_pairs is not None:
            return

        # 1. Prepare Data for PCA (Use all 27 columns)
        feature_cols = [str(i) for i in range(1, 28)]
        # Fill NaNs if any (though loader handles it, safety check)
        data = self.df[feature_cols].fillna(0)
        
        # 2. PCA Projection
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        pca = PCA(n_components=5) # Capture major variance
        data_pca = pca.fit_transform(data_scaled)
        
        self.df['pca_features'] = list(data_pca)
        
        # 3. Split QIM and ASRP indices
        qim_indices = self.df[self.df['운용월'] == 0].index
        asrp_indices = self.df[self.df['운용월'] > 0].index
        
        qim_pca = np.stack(self.df.loc[qim_indices, 'pca_features'].values)
        asrp_pca = np.stack(self.df.loc[asrp_indices, 'pca_features'].values)
        
        # 4. Nearest Neighbor Matching
        # Find closest QIM for each ASRP
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(qim_pca)
        distances, indices = nbrs.kneighbors(asrp_pca)
        
        pairs = []
        for i, asrp_idx in enumerate(asrp_indices):
            qim_idx = qim_indices[indices[i][0]]
            pairs.append({
                'qim_idx': qim_idx,
                'asrp_idx': asrp_idx,
                'qim_month': 0,
                'asrp_month': self.df.loc[asrp_idx, '운용월'],
                'distance': distances[i][0]
            })
            
        self.matched_pairs = pd.DataFrame(pairs)

    def fit_matching_models(self, target_col):
        if self.matched_pairs is None:
            self.perform_matching()
            
        # Prepare Training Data: Delta vs Time
        # We want to predict: Value(t) = Value(0) + Delta(t)
        # So we model Delta(t) ~ f(t, initial_value)
        
        X_train = []
        y_train = []
        
        for _, row in self.matched_pairs.iterrows():
            qim_val = self.df.loc[row['qim_idx'], target_col]
            asrp_val = self.df.loc[row['asrp_idx'], target_col]
            
            if pd.isna(qim_val) or pd.isna(asrp_val):
                continue
                
            delta = asrp_val - qim_val
            time = row['asrp_month']
            
            X_train.append([time, qim_val]) # Features: Time, Initial Value
            y_train.append(delta)
            
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Model 1: Linear Regression on Delta
        # Delta = a * time + b * initial_val + c
        lin_model = LinearRegression()
        lin_model.fit(X_train, y_train)
        
        # Model 2: Random Forest on Delta
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        self.models_matching[target_col] = {
            'Linear_Matching': lin_model,
            'RF_Matching': rf_model
        }

    def predict_matching(self, target_col, future_months):
        if target_col not in self.models_matching:
            self.fit_matching_models(target_col)
            
        # For prediction, we assume an "Average" QIM item
        avg_qim_val = self.df[self.df['운용월'] == 0][target_col].mean()
        
        results = {}
        X_pred = np.array([[m, avg_qim_val] for m in future_months])
        
        for name, model in self.models_matching[target_col].items():
            delta_pred = model.predict(X_pred)
            y_pred = avg_qim_val + delta_pred
            
            # Simple CI (using training std of deltas)
            # This is a simplification; RF has other ways, but keeping it consistent
            # Calculate residuals on training data
            # Re-construct X_train to calculate residuals
            X_train = []
            y_train_true = []
            for _, row in self.matched_pairs.iterrows():
                qim_val = self.df.loc[row['qim_idx'], target_col]
                asrp_val = self.df.loc[row['asrp_idx'], target_col]
                X_train.append([row['asrp_month'], qim_val])
                y_train_true.append(asrp_val - qim_val)
            
            delta_train_pred = model.predict(X_train)
            std = np.std(np.array(y_train_true) - delta_train_pred)
            
            results[name] = (y_pred, y_pred - 1.645*std, y_pred + 1.645*std)
            
        return results

    # --- Phase 1: Full Screening ---
    def calculate_all_trends(self, limits_df=None):
        """
        Iterates through all measurement columns (1-27), calculates linear slope and R2.
        Returns a DataFrame sorted by Normalized Slope (Slope / Spec_Range * 100).
        """
        results = []
        feature_cols = [str(i) for i in range(1, 28)]
        
        X = self.df[['운용월']].values
        
        for col in feature_cols:
            if col not in self.df.columns:
                continue
                
            # Drop NaNs for this column
            valid_mask = self.df[col].notna() & self.df['운용월'].notna()
            if valid_mask.sum() < 2: # Need at least 2 points
                continue
                
            y = self.df.loc[valid_mask, col].values
            X_valid = self.df.loc[valid_mask, ['운용월']].values
            
            # Simple Linear Regression for Screening
            model = LinearRegression()
            model.fit(X_valid, y)
            
            slope = model.coef_[0]
            r2 = model.score(X_valid, y)
            
            # Calculate Variance Ratio (ASRP Var / QIM Var)
            qim_data = self.df[(self.df['운용월'] == 0) & (self.df[col].notna())][col]
            asrp_data = self.df[(self.df['운용월'] > 0) & (self.df[col].notna())][col]
            
            qim_var = qim_data.var() if len(qim_data) > 1 else 0
            asrp_var = asrp_data.var() if len(asrp_data) > 1 else 0
            var_ratio = asrp_var / qim_var if qim_var > 0 else 0
            
            # Calculate Normalized Slope (User Request)
            norm_slope = 0
            spec_range = np.nan
            
            if limits_df is not None:
                item_limit = limits_df[limits_df['Item'] == col]
                if not item_limit.empty:
                    usl = item_limit['USL'].values[0]
                    lsl = item_limit['LSL'].values[0]
                    
                    if not np.isnan(usl) and not np.isnan(lsl):
                        spec_range = usl - lsl
                        if spec_range > 0:
                            # % Change per Month relative to Range
                            norm_slope = (abs(slope) / spec_range) * 100
            
            results.append({
                'Item': col,
                'Slope': slope,
                'Abs_Slope': abs(slope),
                'Norm_Slope': norm_slope, # New Metric
                'Spec_Range': spec_range,
                'R2': r2,
                'Var_Ratio': var_ratio
            })
            
        results_df = pd.DataFrame(results)
        # Sort by Normalized Slope instead of Abs_Slope
        results_df = results_df.sort_values(by='Norm_Slope', ascending=False).reset_index(drop=True)
        
        return results_df
