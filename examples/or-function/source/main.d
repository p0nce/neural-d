import std.math;

void main(string[] args)
{
  //  loadData();


}


struct Neuron
{
    
}


// Type of Layers
// dense, dropout, convolutional, pooling, recurrent layers

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x)
{
    double ex = exp(-x);
    return ex / (ex + 1.0)*(ex + 1.0);
}

void softmax(double[] inoutCoeffients)
{
    // Numerically stable with large exponentials
    size_t N = inoutCoeffients.length;

    double[] temp = new double[N];
    foreach(size_t n; 0..N)
    {
        temp[n] = exp( inoutCoeffients[n] );
    }

    double sum = 0;
    foreach(size_t n; 0..N)
    {
        sum += temp[n];
    }
    foreach(size_t n; 0..N)
    {
        inoutCoeffients[n] = temp[n] / sum;
    }
}

unittest
{
    double[] A = [1.0, 3, 2.5, 5, 4, 2];
    softmax(A);
    double[] expected = [0.011, 0.082, 0.05, 0.605, 0.222, 0.030];
    foreach(n; 0..6)
    {
        assert( abs(A[n] - expected[n]) );
    }
}
