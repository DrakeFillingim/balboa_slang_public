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
                  StructuredBuffer<QuadraticBezier> quadratic_beziers,
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

            float2 sub_pos = float2(pixel_pos.x + xx / float(samples_per_axis) + .5 / float(samples_per_axis),
                                    pixel_pos.y + yy / float(samples_per_axis) + .5 / float(samples_per_axis));

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
                if (s.type == ShapeType.QuadraticBezier)
                {
                    QuadraticBezier b = quadratic_beziers[s.index];
                    in_fill = is_quadratic_bezier_fill(transform_pos, points, s, b);
                    in_stroke = is_quadratic_bezier_stroke(transform_pos, points, s, b);
                }

                if (in_fill)
                {
                    sub_color = sub_color * (1 - s.fill_rgba.w) + (s.fill_rgba * s.fill_rgba.w);
                    // sub_color = s.fill_rgba;
                }
                if (in_stroke)
                {
                    sub_color = sub_color * (1 - s.stroke_rgba.w) + (s.stroke_rgba * s.stroke_rgba.w);
                    // sub_color = s.stroke_rgba;
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

            if (in_polyline_stroke_width(pixel_pos, p0, p1, s.stroke_width))
                return true;
        }
    }

    // rounded corners
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
bool in_polyline_stroke_width(float2 q, float2 p0, float2 p1, float stroke_width)
{
    float l = dot(p1 - p0, q - p0) / length(p1 - p0);
    if (l <= 0 || l >= length(p1 - p0)) return false;
    float2 qprime = p0 + l * (p1 - p0) / length(p1 - p0);
    return length(q - qprime) < stroke_width / 2;
}

// pts_end is exclusive
bool is_quadratic_bezier_stroke(float2 pixel_pos, StructuredBuffer<float2> points, Shape s, QuadraticBezier b)
{
    if (!s.use_stroke) return false;

    for (int i = b.pts_begin; i < b.pts_end - 1; i += 2)
    {
        float2 p0 = points[i];
        float2 p1 = points[i + 1];
        float2 p2 = points[i + 2];
        if (i + 2 > b.pts_end - 1) p2 = points[b.pts_begin];


        if (in_bezier_stroke_width(pixel_pos, p0, p1, p2, s.stroke_width)) return true;
    }

    // rounded corners
    int start = b.pts_begin;
    int end = b.pts_end;
    if (!b.closed) { start += 2; end -= 2; }
    for (int i = start; i < end; i += 2)
    {
        if (length(pixel_pos - points[i]) < s.stroke_width / 2)
            return true;
    }

    return false;
}

bool in_bezier_stroke_width(float2 q, float2 p0, float2 p1, float2 p2, float stroke_width)
{
    float a = dot(p0 - 2 * p1 + p2, 2 * p0 - 4 * p1 + 2 * p2);
    float b = dot(-2 * p0 + 2 * p1, 2 * p0 - 4 * p1 + 2 * p2) + 
              dot(p0 -2 * p1 + p2, 2 * p1 - 2 * p0);
    float c = dot(-2 * p0 + 2 * p1, 2 * p1 - 2 * p0) + 
              dot(p0 - q, 2 * p0 - 4 * p1 + 2 * p2);
    float d = dot(p0 - q, 2 * p1 - 2 * p0);
    float3 roots;
    int num_roots = solve_cubic(a, b, c, d, roots);
    for (int i = 0; i < num_roots; i++)
    {
        float t = roots[i];
        if (t >= 0 && t < 1)
        {
            float2 ftstar = p0 * (1 - t) * (1 - t) + 2 * p1 * t * (1 - t) + p2 * t * t;
            float distance = length(ftstar - q);
            if (distance < stroke_width / 2) return true;
        }
    }
    return false;
}

bool is_quadratic_bezier_fill(float2 pixel_pos, StructuredBuffer<float2> points, Shape s, QuadraticBezier b)
{
    if (!s.use_fill || !b.closed) return false;
    int count = 0;
    for (int i = b.pts_begin; i < b.pts_end; i += 2)
    {
        float2 p0 = points[i];
        float2 p1 = points[i + 1];
        float2 p2 = points[b.pts_begin];
        if (i + 2 < b.pts_end - 1) p2 = points[i + 2];

        float2 q = pixel_pos + float2(0, 0.0001);
        float2 roots = solve_quadratic_bezier(q, p0, p1, p2);

        count += compute_bezier_ray(q, p0, p1, p2, roots);
    }
    if (count != 0) return true;
    return false;
}

float2 solve_quadratic_bezier(float2 q, float2 p0, float2 p1, float2 p2)
{
    float a = p0.y - 2 * p1.y + p2.y;
    float b = -2 * p0.y + 2 * p1.y;
    float c = p0.y - q.y;
    float2 roots = float2(-1, -1);
    if (solve_quadratic(a, b, c, roots)) return roots;
    return float2(-1, -1);
}

int compute_bezier_ray(float2 q, float2 p0, float2 p1, float2 p2, float2 roots)
{
    float s;
    float t;
    int count = 0;

    if (roots.x >= 0 && roots.x < 1)
    {
        t = roots.x;
        s = (1 - t) * (1 - t) * p0.x + 2 * t * (1 - t) * p1.x + t * t * p2.x - q.x;
        if (s >= 0)
        {
            int sign = bezier_increasing(p0, p1, p2, t);
            if (sign > 0) count++;
            if (sign < 0) count--;
        }
    }
    if (roots.y >= 0 && roots.y < 1)
    {
        t = roots.y;
        s = (1 - t) * (1 - t) * p0.x + 2 * t * (1 - t) * p1.x + t * t * p2.x - q.x;
        if (s >= 0)
        {
            int sign = bezier_increasing(p0, p1, p2, t);
            if (sign > 0) count++;
            if (sign < 0) count--;
        }
        
    }
    return count; 
}

// assume t is between 0 and 1 here.
int bezier_increasing(float2 p0, float2 p1, float2 p2, float t)
{
    float2 d1 = -2 * (1 - t) * p0 + 2 * (1 - 2 * t) * p1 + 2 * t * p2;
    return sign(d1.y);
}