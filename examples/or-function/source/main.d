import neural;


import std.random;

float addFunction(float x, float y)
{
    return x * 0.7f + y * 0.3f + 1;
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
        y[n] = 3.14150f * x[n] + uniform(-0.2f, 0.2f);
    }

    auto model = new Sequential();
    model.add( new Dense(2), Shape(2) );
    model.add( new Activation(ActivationFunction.SELU ) );
    model.add( new Dense(1) );

    model.compile(new SGDOptimizer(0.01f), LossFunction.MSE );

    int epochs = 10;
    model.fit(x[0..N_TRAINING], y[0..N_TRAINING], epochs);
}

