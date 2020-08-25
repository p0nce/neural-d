module neural.tensor;

import std.random;

struct Shape
{
    int batchSize           = 1;
    int inputSizeY          = 1;
    int inputSizeX          = 1;
    int inputSizeDimensions = 1;

    // Returns: `true` if the shape is a valid shape. Else it's an unknown shape.
    bool isValid() pure const nothrow @nogc
    {
        // Example: 30 images of 25x25 RGB pixels are (30, 25, 25, 3)
        return (batchSize           >= 1) 
            && (inputSizeX          >= 1)
            && (inputSizeY          >= 1)
            && (inputSizeDimensions >= 1);
    }

    // Returns: `true` if each sample is just one float.
    bool is1D() pure const nothrow @nogc
    {
        return inputSizeX == 1 && inputSizeY == 1 && inputSizeDimensions == 1;
    }

    // Returns: `true` if each sample is an array of float.
    bool is2D() pure const nothrow @nogc
    {
        return inputSizeY == 1 && inputSizeDimensions == 1;
    }

    // Number of scalar number to represent this shape.
    int elemCount() pure const nothrow @nogc
    {
        return batchSize * inputSizeX * inputSizeY * inputSizeDimensions;
    }

    Shape withBatchSize(int batchSize) pure const nothrow @nogc
    {
        Shape s = this;
        s.batchSize = batchSize;
        return s;
    }
}

enum invalidShape = Shape(-1, -1, -1, -1);

unittest
{
    assert(!invalidShape.isValid());
}

struct Tensor
{
public:
    this(Shape shape)
    {
        resize(shape);
    }

    this(float[] shape)
    {
        resize(Shape(cast(int)shape.length, 1, 1, 1));
        _data[] = shape[];
    }

    ~this()
    {
        _data.length = 0;
    }

    @disable this(this); // Use tensorCopy to make a copy

    void resize(Shape shape)
    {
        _data.length = shape.elemCount();
        _shape = shape;
    }

    Shape shape() const
    {
        return _shape;
    }

    void fillWith(float x)
    {
        _data[] = x;
    }

    inout(float)[] rawData() inout
    {
        return _data;
    }

private:
    Shape _shape = Shape(0, 0, 0, 0);
    float[] _data = null;
}

Tensor zeroes(Shape shape)
{
    Tensor t = Tensor(shape);
    t.fillWith(0.0f);
    return t;
}

Tensor ones(Shape shape)
{
    Tensor t = Tensor(shape);
    t.fillWith(1.0f);
    return t;
}

Tensor randomUniform(Shape shape)
{
    Tensor t = Tensor(shape);
    foreach(ref f; t.rawData)
        f = uniform01();
    return t;
}

// Operations on tensor

/// Make a copy of the tensor. `dest` doesn't need to be pre-allocated.
void tensorAssign(ref Tensor dest, ref const(Tensor) source)
{
    dest.resize(source.shape);
    tensorCopy(dest, source);
}

/// Copy data of an existing allocated tensor into another allocated tensor.
void tensorCopy(ref Tensor dest, ref const(Tensor) source)
{
    assert(dest.shape == source.shape);
    dest._data[] = source._data[];
}


private
{
    @disable double nextRandom(ref ulong seed)
    {
        seed = seed * 1584831133568692850 + 8629370730060374025;
        ulong r = seed >> 32;
        seed ^= r;
        return( ( cast(uint)r ) * 2.32830643653869629e-10 );
    }
}

void axpy(float a, float[] x, float[] y) pure nothrow @nogc
{
    for(int i = 0; i < x.length; ++i)
        y[i] += a * x[i];
}