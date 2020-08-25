module neural.activation;

import inteli.math;

enum ActivationFunction
{
    ReLU,   
    SELU, 
}

void applyActivationFunction(ActivationFunction activation, float[] outputs)
{
    final switch(activation) with (ActivationFunction)
    {
        case ReLU:
            foreach(ref x; outputs)
            {
                if (x < 0) x = 0;
            }
            break;

        case SELU:
            foreach(ref x; outputs)
            {
                float alpha = 1.673263242354377f;
                float scale = 1.05070098735548f;
                if (x < 0)
                {
                    x = alpha * _mm_exp_ss(x) - 1.0f;
                } 
            }
            break;

            // SELU derivative
            //return self.scale * np.where(x >= 0.0, 1, self.alpha * np.exp(x))
    }
}
       

void softmax(double[] inoutCoeffients)
{
    // Numerically stable with large exponentials
    size_t N = inoutCoeffients.length;

    double[] temp = new double[N];
    foreach(size_t n; 0..N)
    {
        temp[n] = _mm_exp_ss( inoutCoeffients[n] );
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
    import std.math;
    
    double[] A = [1.0, 3, 2.5, 5, 4, 2];
    softmax(A);
    double[] expected = [0.011, 0.082, 0.05, 0.605, 0.222, 0.030];
    foreach(n; 0..6)
    {
        assert( abs(A[n] - expected[n]) );
    }
}
