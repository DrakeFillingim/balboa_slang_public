import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid: SV_DispatchThreadID,
                  uniform float3 background,
                  uniform int samples_per_axis,
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
    float2 pixel_pos = float2(tid.x, tid.y);

    image[outPos] = float4(background, 1.0);

    float4 color_avg = float4(0);
    for (int xx = 0; xx < samples_per_axis; xx++)
    {
        for (int yy = 0; yy < samples_per_axis; yy++)
        {
            float4 sub_color = float4(background, 1);

            float2 sub_pos = float2(pixel_pos.x + xx / float(samples_per_axis),
                                    pixel_pos.y + yy / float(samples_per_axis));

            for (int i = num_shapes - 1; i >= 0; i--)
            {
                Shape s = shapes[i];

                float3x1 p = mul(s.world_to_obj, float3x1(sub_pos.x, sub_pos.y, 1));
                float2 transform_pos = float2(p._m00, p._m10);

                bool in_fill = false;
                bool in_stroke = false;

                if (s.type == ShapeType.Circle)
                {
                    in_fill = is_circle_fill(transform_pos, s, circles[s.index]);
                    in_stroke = is_circle_stroke(transform_pos, s, circles[s.index]);
                }
                if (s.type == ShapeType.Polyline)
                {
                    Polyline polyline = polylines[s.index];
                    in_fill = is_polyline_fill(transform_pos, points, s, polyline);
                    in_stroke = is_polyline_stroke(transform_pos, points, s, polyline);
                }

                if (in_fill)
                {
                    sub_color = s.fill_rgba;
                }
                if (in_stroke)
                {
                    sub_color = s.stroke_rgba;
                }
            }
            color_avg += sub_color;
        }
    }
    image[outPos] = color_avg / (samples_per_axis * samples_per_axis);
}

bool is_circle_fill(float2 pixel_pos, Shape s, Circle c)
{
    if (s.use_fill && length(pixel_pos - c.center) < c.radius) return true;
    return false;
}

bool is_circle_stroke(float2 pixel_pos, Shape s, Circle c)
{
    if (s.use_stroke &&
        length(pixel_pos - c.center) < c.radius + s.stroke_width / 2 &&
        length(pixel_pos - c.center) > c.radius - s.stroke_width / 2)
        return true;
    return false;
}

// Checks if a pixel at pixel_pos is a fill pixel using the non-zero test.
bool is_polyline_fill(float2 pixel_pos, StructuredBuffer<float2> points, Shape s, Polyline p)
{
    if (!s.use_fill) return false;
    int count = 0;
    for (int i = p.pts_begin; i < p.pts_end; i++)
    {
        float2 p0 = points[i];
        float2 p1 = points[p.pts_begin];
        if (i != p.pts_end - 1) p1 = points[i + 1];
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

bool is_polyline_stroke(float2 pixel_pos, StructuredBuffer<float2> points, Shape s, Polyline p)
{
    if (!s.use_stroke) return false;
    
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

        if (in_stroke_width(pixel_pos, p0, p1, s.stroke_width))
            return true;
        }
    }

    int start = p.pts_begin;
    if (!p.closed) start++;
    for (int i = start; i < end_point; i++)
    {
        if (length(pixel_pos - points[i]) < s.stroke_width / 2)
            return true;
    }

    return false;
}

// Checks if a pixel located at q is a stroke pixel for polylines.
bool in_stroke_width(float2 q, float2 p0, float2 p1, float stroke_width)
{
    float l = dot(p1 - p0, q - p0) / length(p1 - p0);
    if (l <= 0 || l >= length(p1 - p0)) return false;
    float2 qprime = p0 + l * (p1 - p0) / length(p1 - p0);
    return length(q - qprime) < stroke_width / 2;
}