from numba import jit
import numpy as np

@jit(nopython=True)
def shuffle2D(X):
    idx = np.where(~np.isnan(X[:,0]))[0]
    T = len(idx)
    # Fisher-Yates shuffle
    for i in range(T-1, 0, -1):
        # Pick random integer 0 <= j <= i
        j = np.random.randint(0, i + 1) 
        n, m = idx[i], idx[j]
        for k in range(X.shape[1]):
            X[n,k], X[m,k] = X[m,k], X[n,k]

@jit(nopython=True)
def nearest_centroid_accuracy(train_data, valid_data):
    Q, M, T = train_data.shape
    accuracy = np.zeros(T)
    
    class_means = np.zeros((Q, T))
    for q in range(Q):
        for t in range(T):
            sum_val = 0.0
            for m in range(M):
                sum_val += train_data[q, m, t]
            class_means[q, t] = sum_val / M
            
    for t in range(T):
        correct_count = 0
        for q_val in range(Q):
            for m in range(M):
                val_point = valid_data[q_val, m, t]
                
                min_dist = np.inf
                best_q = -1
                for q_train in range(Q):
                    dist = abs(class_means[q_train, t] - val_point)
                    if dist < min_dist:
                        min_dist = dist
                        best_q = q_train
                        
                if best_q == q_val:
                    correct_count += 1
                    
        accuracy[t] = correct_count / (Q * M)
        
    return accuracy

def get_noise_covariance(X, trialX, N_samples, simultaneous=False, type='pooled'):
    n_features = X.shape[0]
    
    if type == 'none':
        return np.zeros((n_features, n_features))
        
    if not simultaneous:
        X_expanded = np.expand_dims(X, axis=0)
        residuals = trialX - X_expanded
        
        SSnoise = np.nansum(residuals**2, axis=0)
        
        axes_to_sum = tuple(range(1, SSnoise.ndim))
        if axes_to_sum:
            SSnoiseSumOverT = np.sum(SSnoise, axis=axes_to_sum)
        else:
            SSnoiseSumOverT = SSnoise
            
        if axes_to_sum:
            N_samples_avg = N_samples.copy().astype(float)
            N_samples_avg[N_samples_avg == 0] = np.nan
            N_samples_avg = np.nanmean(N_samples_avg, axis=axes_to_sum)
        else:
            N_samples_avg = N_samples.astype(float)
        
        if type == 'pooled':
            variance = SSnoiseSumOverT / N_samples_avg
            Cnoise = np.diag(variance)
        elif type == 'averaged':
            variance_per_cond = SSnoise / N_samples
            if axes_to_sum:
                variance = np.nansum(variance_per_cond, axis=axes_to_sum)
            else:
                variance = variance_per_cond
            Cnoise = np.diag(variance)
        else:
            Cnoise = np.zeros((n_features, n_features))
    else:
        if type == 'pooled':
            X_expanded = np.expand_dims(X, axis=0)
            Xnoise = trialX - X_expanded
            
            Xnoise = np.rollaxis(Xnoise, 1, 0)
            Xnoise = Xnoise.reshape(n_features, -1)
            
            valid_cols = ~np.isnan(Xnoise[0, :])
            Xnoise = Xnoise[:, valid_cols]
            
            SSnoise = np.dot(Xnoise, Xnoise.T)
            Cnoise = SSnoise / Xnoise.shape[1]
            
            dimProd = np.prod(X.shape[1:]) if len(X.shape) > 1 else 1
            Cnoise = Cnoise * dimProd
        elif type == 'averaged':
            print('Averaged noise covariance computation is not yet implemented')
            print('for the simultaneously recorded data. Returning the pooled')
            print('noise covariance matrix instead.')
            Cnoise = get_noise_covariance(X, trialX, N_samples, simultaneous=True, type='pooled')
        else:
            Cnoise = np.zeros((n_features, n_features))
            
    return Cnoise

@jit(nopython=True)
def denoise_mask(mask, n_consecutive):
    subseq = 0
    N = mask.shape[0]
        
    for n in range(N):
        if mask[n] == 1:
            subseq += 1
        else:
            if subseq < n_consecutive:
                for k in range(n-subseq,n):
                    mask[k] = 0
                    
    return mask