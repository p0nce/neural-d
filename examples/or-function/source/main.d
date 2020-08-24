import neural;


import std.random;

float addFunction(float x, float y)
{
    return x * 0.7f + y * 0.3f + 1;
}

void main(string[] args)
{
    auto nn = (new NeuralNetwork(2, 1))
         //     .addLayer( new LayerData(2, 1, ) )
              .addLayer( new LayerLinear(2, 1, ) )
              .addLayer( new LayerReLU() );
}