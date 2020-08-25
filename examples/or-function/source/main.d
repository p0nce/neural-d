import neural;


import std.random;

float addFunction(float x, float y)
{
    return x * 0.7f + y * 0.3f + 1;
}


double nextRandom(ref ulong seed)
{
    seed = seed * 1584831133568692850 + 8629370730060374025;
    ulong r = seed >> 32;
    seed ^= r;
    return( ( cast(uint)r ) * 2.32830643653869629e-10 );
}

void main(string[] args)
{

    auto model = new Sequential();
    model.add( new Input(Shape(2));
    model.add( new Dense(1) );
 //   model.add( new Activation(ActivationFunction.SELU ) );
   // model.initialize();
   // nn.train();
}

