import neural;


import std.random;

float functionToApproximate(float x)
{
    return 3.14150f * x + 2 +  uniform(-0.1f, 0.1f);
}

void main(string[] args)
{
    int N_DATASET = 2000;
    int N_TRAINING = 1000;
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

    int epochs = 100000;
    int minibatch = 32;

    // Compute MSE and display it
    void displayMSE()
    {
        Tensor input  = Tensor(x[N_TRAINING..N_DATASET]);
        Tensor expected = Tensor(y[N_TRAINING..N_DATASET]);
        Tensor output;
        model.predictBatch(input, output);
        double MSE = 0;

        for(int n = 0; n < N_TEST; ++n)
        {
            MSE += ((expected.rawData[n] - output.rawData[n]) * (expected.rawData[n] - output.rawData[n]));
        }
        import std.stdio;
        writefln("MSE = %s", MSE);
    }

    foreach(epoch; 0..1000)
    {
        model.train(Tensor(x[0..N_TRAINING]), 
                    Tensor(y[0..N_TRAINING]), 
                    minibatch,
                    1);
        displayMSE();
    }
    
}

