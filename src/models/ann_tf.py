
import tensorflow.keras as keras
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

class ANNModel:

    def __init__(self, layers = None) -> None:
       
        self.pre_poly = MinMaxScaler((1, 2))
        self.poly     = PolynomialFeatures(5)
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.layers = layers if layers else [256, 512]

        

    def fit(self, x_train, y_train):

        x_t = self.pre_poly.fit_transform(x_train)
        x_t = self.poly.fit_transform(x_t)  
        x_t = self.x_scaler.fit_transform(x_t)

        y_t = self.y_scaler.fit_transform(y_train)

        activation = 'relu'
        output='linear'
        epochs = 1000
        loss="mse" 
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)


        inputs = keras.Input(shape=(x_t.shape[1],), name='parameters')
        
        lay = inputs

        for n in self.layers:
            lay = keras.layers.Dense(n, activation=activation, 
               kernel_regularizer=keras.regularizers.L2(0.000001), 
               activity_regularizer=keras.regularizers.L2(0.001))(lay)

        outputs = keras.layers.Dense(y_t.shape[1], activation=output, 
            kernel_regularizer=keras.regularizers.L2(0.000001))(lay)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            loss=loss,
            optimizer=optimizer)
        
        self.history = self.model.fit(x_t, y_t, 
                    epochs = epochs, 
                    batch_size= 64, 
                    verbose = 0)


    def predict(self, x_test):
        return self.y_scaler.inverse_transform(
            self.model.predict(self.x_scaler.transform(
                self.poly.transform(self.pre_poly.transform(x_test)))))



