import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

class TrendPredictor:
    def __init__(self, df):
        self.df = df
        self.models_population = {}
        self.model_metrics = {}
        self.models_matching = {}
        self.matched_pairs = None

    def fit_population_models(self, target_col):
        X = self.df[['운용월']].values
        y = self.df[target_col].values
        
        self.models_population[target_col] = {}
        self.model_metrics[target_col] = {}
        
        # 1. Linear Regression
        model_lin = LinearRegression()
        model_lin.fit(X, y)
        self.models_population[target_col]['Linear'] = model_lin
        self.model_metrics[target_col]['Linear'] = np.sqrt(mean_squared_error(y, model_lin.predict(X)))
        
        # 2. Polynomial Regression (Degree 2)
        model_poly = make_pipeline(PolynomialFeatures(2), LinearRegression())
        model_poly.fit(X, y)
        self.models_population[target_col]['Polynomial'] = model_poly
        self.model_metrics[target_col]['Polynomial'] = np.sqrt(mean_squared_error(y, model_poly.predict(X)))
        
        # 3. Bayesian Ridge
        model_bayes = make_pipeline(PolynomialFeatures(2), BayesianRidge())
        model_bayes.fit(X, y)
        self.models_population[target_col]['Bayesian'] = model_bayes
        self.model_metrics[target_col]['Bayesian'] = np.sqrt(mean_squared_error(y, model_bayes.predict(X)))
        
        # 4. Gaussian Process
        kernel = C(1.0, (1e-3, 1e3)) * RBF(100, (1e-2, 1e4))
        model_gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=0.1)
        model_gp.fit(X, y)
        self.models_population[target_col]['GaussianProcess'] = model_gp
        self.model_metrics[target_col]['GaussianProcess'] = np.sqrt(mean_squared_error(y, model_gp.predict(X)))
        
        # 5. SVR
        model_svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
        model_svr.fit(X, y)
        self.models_population[target_col]['SVR'] = model_svr
        self.model_metrics[target_col]['SVR'] = np.sqrt(mean_squared_error(y, model_svr.predict(X)))
        
        # 6. Neural Network (MLP)
        model_nn = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42))
        model_nn.fit(X, y)
        self.models_population[target_col]['NeuralNetwork'] = model_nn
        self.model_metrics[target_col]['NeuralNetwork'] = np.sqrt(mean_squared_error(y, model_nn.predict(X)))

    def predict_population(self, target_col, future_months):
        if target_col not in self.models_population:
            self.fit_population_models(target_col)
            
        X_pred = np.array(future_months).reshape(-1, 1)
        results = {}
        
        for name, model in self.models_population[target_col].items():
            if name in ['Bayesian', 'GaussianProcess']:
                # Bayesian and GP provide std
                if name == 'Bayesian':
                    # BayesianRidge inside pipeline
                    y_pred, y_std = model.predict(X_pred, return_std=True)
                else:
                    y_pred, y_std = model.predict(X_pred, return_std=True)
                results[name] = (y_pred, y_pred - 1.645*y_std, y_pred + 1.645*y_std)
            else:
                y_pred = model.predict(X_pred)
                # For others, use training residual std for CI
                y_hat = model.predict(self.df[['운용월']].values)
                std = np.std(self.df[target_col].values - y_hat)
                results[name] = (y_pred, y_pred - 1.645*std, y_pred + 1.645*std)
                
        return results

    def perform_matching(self):
        feature_cols = [str(i) for i in range(1, 28)]
        data = self.df[feature_cols].fillna(self.df[feature_cols].mean()).fillna(0)
        
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        pca = PCA(n_components=min(5, len(feature_cols)))
        data_pca = pca.fit_transform(data_scaled)
        
        qim_idx = self.df[self.df['Dataset'] == 'QIM'].index
        asrp_idx = self.df[self.df['Dataset'] == 'ASRP'].index
        
        if len(qim_idx) == 0 or len(asrp_idx) == 0:
            self.matched_pairs = pd.DataFrame()
            return

        qim_pca = data_pca[qim_idx]
        asrp_pca = data_pca[asrp_idx]
        
        nbrs = NearestNeighbors(n_neighbors=1).fit(qim_pca)
        distances, indices = nbrs.kneighbors(asrp_pca)
        
        pairs = []
        for i, a_idx in enumerate(asrp_idx):
            q_idx = qim_idx[indices[i][0]]
            pairs.append({
                'qim_idx': q_idx, 'asrp_idx': a_idx,
                'qim_month': 0, 'asrp_month': self.df.loc[a_idx, '운용월'],
                'distance': distances[i][0]
            })
        self.matched_pairs = pd.DataFrame(pairs)

    def predict_matching(self, target_col, future_months):
        if self.matched_pairs is None:
            self.perform_matching()
        if self.matched_pairs.empty:
            return {}
            
        # Implementation of simple matching trend (Average Delta)
        # Re-using logic from backup for completeness
        X_train = []
        y_train = []
        for _, row in self.matched_pairs.iterrows():
            q_val = self.df.loc[row['qim_idx'], target_col]
            a_val = self.df.loc[row['asrp_idx'], target_col]
            X_train.append([row['asrp_month']])
            y_train.append(a_val - q_val)
            
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        avg_qim_val = self.df[self.df['Dataset'] == 'QIM'][target_col].mean()
        X_pred = np.array(future_months).reshape(-1, 1)
        delta_pred = model.predict(X_pred)
        
        y_pred = avg_qim_val + delta_pred
        std = np.std(np.array(y_train) - model.predict(X_train))
        
        return {'Matching_Trend': (y_pred, y_pred - 1.645*std, y_pred + 1.645*std)}

    def calculate_all_trends(self, limits_df=None):
        results = []
        feature_cols = [str(i) for i in range(1, 28)]
        X = self.df[['운용월']].values
        
        for col in feature_cols:
            if col not in self.df.columns:
                continue
                
            y = self.df[col].values
            model = LinearRegression()
            model.fit(X, y)
            
            slope = model.coef_[0]
            r2 = model.score(X, y)
            
            qim_var = self.df[self.df['Dataset'] == 'QIM'][col].var()
            asrp_var = self.df[self.df['Dataset'] == 'ASRP'][col].var()
            var_ratio = asrp_var / qim_var if qim_var > 0 else 0
            
            # Use USL/LSL for Norm_Slope normalization if available
            norm_factor = 1.0
            if limits_df is not None:
                limit = limits_df[limits_df['Item'] == col]
                if not limit.empty:
                    usl = limit['USL'].values[0]
                    lsl = limit['LSL'].values[0]
                    if pd.notna(usl) and pd.notna(lsl):
                        norm_factor = abs(usl - lsl)
            
            norm_slope = slope / norm_factor
            
            results.append({
                'Item': col,
                'Slope': slope,
                'Norm_Slope': norm_slope,
                'R2': r2,
                'Var_Ratio': var_ratio
            })
            
        return pd.DataFrame(results).sort_values('Norm_Slope', ascending=False)
