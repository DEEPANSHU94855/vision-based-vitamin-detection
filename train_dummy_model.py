import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def train_dummy_model():
    print("Generating structured mock dataset...")
    # 6 target classes: Vitamin A, B12, C, D, Iron/Anaemia, Healthy
    classes = ['Vitamin A Deficiency', 'Vitamin B12 Deficiency', 'Vitamin C Deficiency', 
               'Vitamin D Deficiency', 'Iron/Anaemia', 'Healthy']
    
    X = []
    y = []
    # Generate 500 samples for better hyperplane margin
    for _ in range(500):
        cls = np.random.choice(classes)
        # Base vector (random normal centered around 0.5)
        vec = np.random.normal(loc=0.5, scale=0.1, size=78)
        
        if cls == 'Iron/Anaemia':
            # High pallor index (index 31), Low redness (R mean is index 28)
            vec[31] = np.random.normal(loc=0.9, scale=0.05)
            vec[28] = np.random.normal(loc=0.3, scale=0.05)
        elif cls == 'Vitamin B12 Deficiency':
            # Beefy red tongue: High redness (index 28), high a* (index 24)
            vec[28] = np.random.normal(loc=0.9, scale=0.05)
            vec[24] = np.random.normal(loc=0.8, scale=0.05)
        elif cls == 'Vitamin C Deficiency':
            # Bleeding spots: High contrast/variance (GLCM indices 32-35)
            vec[32:36] = np.random.normal(loc=0.8, scale=0.1, size=4)
        elif cls == 'Vitamin A Deficiency':
            # Eye issues (Bitot's spots): White/foamy, low saturation (indices 16-20)
            vec[16:20] = np.random.normal(loc=0.2, scale=0.1, size=4)
        elif cls == 'Vitamin D Deficiency':
            # General low values / specific skin tones
            vec[20:24] = np.random.normal(loc=0.4, scale=0.1, size=4)
        elif cls == 'Healthy':
            # Balanced normal ranges
            vec[28] = np.random.normal(loc=0.6, scale=0.05) 
            vec[31] = np.random.normal(loc=0.4, scale=0.05)
            
        X.append(vec)
        y.append(cls)
        
    X = np.array(X)
    y = np.array(y)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Training One-Versus-One SVM with RBF kernel...")
    # decision_function_shape='ovo' explicitly sets One-Versus-One Multiclass
    svm_model = SVC(kernel='rbf', decision_function_shape='ovo', probability=True, random_state=42)
    svm_model.fit(X_scaled, y)
    
    print("Saving model and scaler...")
    with open("model.pkl", "wb") as f:
        pickle.dump(svm_model, f)
        
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    print("Mock model saved successfully as 'model.pkl' and 'scaler.pkl'.")

if __name__ == "__main__":
    train_dummy_model()
