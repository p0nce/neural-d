import neural;

import std.stdio;
import std.random;
import std.math;
import std.file;

Tensor loadImageFile(string path)
{
    ubyte[] contents = cast(ubyte)[] std.file.read(path);

}


void main(string[] args)
{
    int N_DATASET = 2000;
    int N_TRAINING = 1000;
    int N_TEST = N_DATASET - N_TRAINING;

    float[] x = new float[2*N_DATASET];
    float[] y = new float[N_DATASET];

    for(int n = 0; n < N_DATASET; ++n)
    {
        x[2*n] = uniform(0.0f, 1.0f);
        x[2*n+1] = uniform(0.0f, 1.0f);
        y[n] = orFunction(x[2*n], x[2*n+1]);
    }

    auto model = new Sequential();
    model.add( new Dense(2), Shape(2) );
    model.add( new Activation(ActivationFunction.SELU ) );
    model.add( new Dense(2) );
    model.add( new Activation(ActivationFunction.SELU ) );
    model.add( new Dense(1) );

    model.summary();

    // Compute MSE and display it
    void displayMSE()
    {
        Tensor input  = Tensor(x[2*N_TRAINING..2*N_DATASET], Shape(N_TEST, 2));
        Tensor expected = Tensor(y[N_TRAINING..N_DATASET], Shape(N_TEST));
        Tensor output;
        model.predictBatch(input, output);
        double MSE = 0;

        for(int n = 0; n < N_TEST; ++n)
        {
            MSE += ((expected.rawData[n] - output.rawData[n]) * (expected.rawData[n] - output.rawData[n]));
        }
        import std.stdio;
        MSE /= N_TEST;
        writefln("MSE = %s", sqrt(MSE));
    }

    model.compile(new SGDOptimizer(0.05f), LossFunction.MSE );

    foreach(epoch; 0..100000)
    {
        int minibatch = 32;
        int offset = epoch % minibatch;
        model.train(Tensor(x[2*offset..2*(N_TRAINING+offset)], Shape(N_TRAINING, 2)), 
                    Tensor(y[offset..N_TRAINING+offset]), 
                    minibatch,
                    1);
        displayMSE();
    }    
}

