import neural;


import std.random;

float functionToApproximate(float x)
{
    return 3.14150f * x + uniform(-0.2f, 0.2f);
}

void main(string[] args)
{
    int N_DATASET = 200;
    int N_TRAINING = 100;
    int N_TEST = N_DATASET - N_TRAINING;

    float[] x = new float[N_DATASET];
    float[] y = new float[N_DATASET];

    for(int n = 0; n < N_DATASET; ++n)
    {
        x[n] = uniform(-1, 1);
        y[n] = functionToApproximate(x[n]);
    }

    auto model = new Sequential();
    model.add( new Dense(2), Shape(2) );
    model.add( new Activation(ActivationFunction.SELU ) );
    model.add( new Dense(1) );

    model.summary();

    model.compile(new SGDOptimizer(0.01f), LossFunction.MSE );

    int epochs = 10;
    int minibatch = 32;

    model.train(Tensor(x[0..N_TRAINING]), 
                Tensor(y[0..N_TRAINING]), 
                minibatch,
                epochs);
}

  //  model.evaluate()
/+
model.computeAccuracy(Tensor(x[0..N_TRAINING]), 
                Tensor(y[0..N_TRAINING])

    model.


}



def compute_accuracy(self, x_val, y_val):
    '''
        This function does a forward pass of x, then checks if the indices
        of the maximum value in the output equals the indices in the label
        y. Then it sums over each prediction and calculates the accuracy.
    '''
    predictions = []

    for x, y in zip(x_val, y_val):
        output = self.forward_pass(x)
        pred = np.argmax(output)
        predictions.append(pred == y)
    
    summed = sum(pred for pred in predictions) / 100.0
    return np.average(summed)
+/