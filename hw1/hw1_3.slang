import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform int /*samples_per_axis*/, // ignore for now
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Circle> circles,
                  StructuredBuffer<Polyline> polylines,
                  StructuredBuffer<QuadraticBezier> /*quadratic_beziers*/, // ignore for now
                  StructuredBuffer<float2> points,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    uint2 outPos = uint2(tid.x, height - 1 - tid.y);
    float2 pixel_pos = float2(tid.x + .5, tid.y + .5);

    image[outPos] = float4(background, 1.0);
    // TODO: your code here
    for (int i = shapes.getCount() - 1; i >= 0; i--)
    {
        Shape s = shapes[i];
        if (s.type == ShapeType.Circle)
        {
            DrawCircle(outPos, pixel_pos, s, circles[s.index], image);
        }
        if (s.type == ShapeType.Polyline)
        {
            DrawPolyline(outPos, pixel_pos, s, polylines[s.index], points, image);
        }
    }
}

void DrawCircle(uint2 outPos, float2 pixel_pos, Shape s, Circle c, RWTexture2D<float4> image)
{
    if (s.use_fill && length(pixel_pos - c.center) < c.radius)
        image[outPos] = s.fill_rgba;
    if (s.use_stroke && 
        length(pixel_pos - c.center) < c.radius + s.stroke_width / 2 &&
        length(pixel_pos - c.center) > c.radius - s.stroke_width / 2)
        image[outPos] = s.stroke_rgba;

}

void DrawPolyline(uint2 outPos, float2 pixel_pos, Shape s, Polyline p, 
                  StructuredBuffer<float2> points,
                  RWTexture2D<float4> image)
{
    // fill color
    if (p.closed && s.use_fill && is_fill(pixel_pos, points,
                                          p.pts_begin, p.pts_end))
        image[outPos] = float4(s.fill_rgba);

    int end_point = p.pts_end;
    if (!p.closed) end_point--;

    // line/stroke color
    if (s.use_stroke)
    {
        for (int i = p.pts_begin; i < end_point; i++)
        {
            float2 p0 = points[i];
            float2 p1 = points[p.pts_begin];
            if (i != p.pts_end - 1) p1 = points[i + 1];

            if (is_stroke(pixel_pos, p0, p1, s.stroke_width))
                image[outPos] = s.stroke_rgba;
        }
    }

    // rounded corners
    if (s.use_stroke)
    {
        int start = p.pts_begin;
        if (!p.closed) start++;
        for (int i = start; i < end_point; i++)
        {
            if (length(pixel_pos - points[i]) < s.stroke_width / 2)
                image[outPos] = s.stroke_rgba;
        }
    }
}

// Checks if a pixel at pixel_pos is a fill pixel using the non-zero test.
bool is_fill(float2 pixel_pos, StructuredBuffer<float2> points, int start, int end)
{
    uint count = 0;
    for (int i = start; i < end; i++)
    {
        float2 p0 = points[i];
        float2 p1 = points[start];
        if (i != end - 1) p1 = points[i + 1];
        if (compute_ray(pixel_pos, p0, p1))
        {
            if (p0.y < p1.y) count++;
            else if (p0.y > p1.y) count--;
        }
    }
    if (count != 0) return true;
    return false;
}

// Checks if a pixel located at q intersects with a line from p0 to p1.
bool compute_ray(float2 q, float2 p0, float2 p1)
{
    if (p0.y == p1.y) return false;
    float t = (q.y - p0.y) / (p1.y - p0.y);
    if (t < 0 || t > 1) return false;
    float s = p0.x + t * (p1.x - p0.x) - q.x;
    return s >= 0;
}

// Checks if a pixel located at q is a stroke pixel.
bool is_stroke(float2 q, float2 p0, float2 p1, float stroke_width)
{
    float l = dot(p1 - p0, q - p0) / length(p1 - p0);
    if (l <= 0 || l >= length(p1 - p0)) return false;
    float2 qprime = p0 + l * (p1 - p0) / length(p1 - p0);
    return length(q - qprime) < stroke_width / 2;
}