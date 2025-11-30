import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline

class TrendPredictor:
    def __init__(self, df):
        self.df = df
        self.models_population = {}
        self.models_matching = {}
        self.matched_pairs = None

    # --- Method A: Population Trend ---
    def fit_population_models(self, target_col):
        X = self.df[['운용월']].values
        y = self.df[target_col].values
        
        # Model 1: Polynomial Regression (Degree 2)
        poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        poly_model.fit(X, y)
        
        # Model 2: Bayesian Ridge (Degree 2 for consistency)
        bayes_model = make_pipeline(PolynomialFeatures(2), BayesianRidge())
        bayes_model.fit(X, y)
        
        self.models_population[target_col] = {
            'Polynomial': poly_model,
            'Bayesian': bayes_model
        }

    def predict_population(self, target_col, future_months):
        if target_col not in self.models_population:
            self.fit_population_models(target_col)
            
        X_pred = np.array(future_months).reshape(-1, 1)
        results = {}
        
        for name, model in self.models_population[target_col].items():
            if name == 'Bayesian':
                y_pred, y_std = model.predict(X_pred, return_std=True)
                results[name] = (y_pred, y_pred - 1.645*y_std, y_pred + 1.645*y_std)
            else:
                y_pred = model.predict(X_pred)
                # Simple CI for Poly (using residual std)
                y_hat = model.predict(self.df[['운용월']].values)
                std = np.std(self.df[target_col].values - y_hat)
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
    def calculate_all_trends(self):
        """
        Iterates through all measurement columns (1-27), calculates linear slope and R2.
        Returns a DataFrame sorted by absolute slope (magnitude of change).
        """
        results = []
        feature_cols = [str(i) for i in range(1, 28)]
        
        X = self.df[['운용월']].values
        
        for col in feature_cols:
            if col not in self.df.columns:
                continue
                
            y = self.df[col].values
            
            # Simple Linear Regression for Screening
            model = LinearRegression()
            model.fit(X, y)
            
            slope = model.coef_[0]
            r2 = model.score(X, y)
            
            # Calculate Variance Ratio (ASRP Var / QIM Var)
            qim_var = self.df[self.df['운용월'] == 0][col].var()
            asrp_var = self.df[self.df['운용월'] > 0][col].var()
            var_ratio = asrp_var / qim_var if qim_var > 0 else 0
            
            results.append({
                'Item': col,
                'Slope': slope,
                'Abs_Slope': abs(slope),
                'R2': r2,
                'Var_Ratio': var_ratio
            })
            
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='Abs_Slope', ascending=False).reset_index(drop=True)
        
        return results_df
