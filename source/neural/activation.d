/*
 * Copyright: 2020 Guillaume Piolat.
 * License: $(LINK2 http://www.apache.org/licenses/LICENSE-2.0, Apache License Version 2.0)
 */
module neural.activation;

import inteli.math;

enum ActivationFunction
{
    SIGMOID,
    RELU,
    LEAKY_RELU,
    SELU, 
}

float evalActivationFunction(ActivationFunction activation, float x)
{
    final switch(activation) with (ActivationFunction)
    {
        case SIGMOID:
        {
            return 1.0f / (1.0f + _mm_exp_ss(x));
        }

        case RELU:
        {
            if (x < 0)
                return 0;
            else
                return x;
        }

        case LEAKY_RELU:
        {
            if (x < 0)
                return 0.3f * x;
            else
                return x;
        }

        case SELU:
        {
            float alpha = 1.673263242354377f;
            float scale = 1.05070098735548f;
            if (x < 0)
                return scale * alpha * (_mm_exp_ss(x) - 1.0f);
            else
                return scale * x;
        }        
    }
}

void applyActivationFunction(ActivationFunction activation, float[] outputs)
{
    foreach(ref x; outputs)
    {
        x = evalActivationFunction(activation, x);
    }
}
       
float evalActivationFunctionDerivative(ActivationFunction activation, float x)
{
    final switch(activation) with (ActivationFunction)
    {
        case SIGMOID:
        {
            float sigx = 1.0f / (1.0f + _mm_exp_ss(x));
            return sigx * (1 - sigx);
        }

        case RELU:
        {
            if (x < 0)
                return 0;
            else
                return 1;
        }

        case LEAKY_RELU:
        {
            if (x < 0)
                return 0.3f;
            else
                return 1;
        }

        case SELU:
        {
            float alpha = 1.673263242354377f;
            float scale = 1.05070098735548f;
            if (x < 0)
                return scale * alpha * _mm_exp_ss(x);
            else
                return scale; 
        }        
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
