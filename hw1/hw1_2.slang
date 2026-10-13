import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  StructuredBuffer<float2> polyline,
                  uniform int num_polyline,
                  uniform bool is_closed,
                  uniform bool use_fill_color,
                  uniform float3 fill_color,
                  uniform bool use_stroke_color,
                  uniform float3 stroke_color,
                  uniform float stroke_width,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    uint2 outPos = uint2(tid.x, height - 1 - tid.y);
    float2 pixel_pos = float2(tid.x +.5, tid.y + .5 + .01);

    image[outPos] = float4(1.0);
    // TODO: your code here

    // fill color
    if (is_closed && use_fill_color && is_fill(pixel_pos, polyline))
        image[outPos] = float4(fill_color, 1);

    // line/stroke color
    if (use_stroke_color)
    {
        int polyCount = polyline.getCount();
        if (!is_closed) polyCount--;
        for (int i = 0; i < polyCount; i++)
        {
            float2 p0 = polyline[i];
            float2 p1 = polyline[(i + 1) % polyline.getCount()];
            if (is_stroke(pixel_pos, p0, p1, stroke_width))
                image[outPos] = float4(stroke_color, 1);
        }
    }

    // round corners
    // stroke width / 2 for radius?
    int start = 0;
    int polyCount = polyline.getCount();
    if (!is_closed) { start = 1; polyCount--; }
    for (int i = start; i < polyCount; i++)
    {
        if (length(pixel_pos - polyline[i]) < stroke_width / 2)
            image[outPos] = float4(stroke_color, 1);
    }
}

bool is_stroke(float2 q, float2 p0, float2 p1, float stroke_width)
{
    float l = dot(p1 - p0, q - p0) / length(p1 - p0);
    if (l <= 0 || l >= length(p1 - p0)) return false;
    float2 qprime = p0 + l * (p1 - p0) / length(p1 - p0);
    return length(q - qprime) < stroke_width / 2;
}

bool is_fill(float2 pixel_pos, StructuredBuffer<float2> polyline)
{
    uint count = 0;
    for (int i = 0; i < polyline.getCount(); i++)
    {
        float2 p0 = polyline[i];
        float2 p1 = polyline[(i + 1) % polyline.getCount()];
        if (compute_ray(pixel_pos, p0, p1))
        {
            if (p0.y < p1.y) count++;
            else if (p0.y > p1.y) count--;
        }
    }
    if (count != 0) return true;
    return false;
}

bool compute_ray(float2 q, float2 p0, float2 p1)
{
    if (p0.y == p1.y) return false;
    float t = (q.y - p0.y) / (p1.y - p0.y);
    if (t < 0 || t > 1) return false;
    float s = p0.x + t * (p1.x - p0.x) - q.x;
    return s >= 0;
}