import hw3_common;

static const float PI = 3.14;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(background, 1);

    float aspect = float(width) / height;
    float2 tid_center = tid + float2(0.5, 0.5);
    float2 pixel_center = 2.0 * (float2(tid_center) / float2(width, height) - float2(0.5, 0.5));
    pixel_center.y *= -1;
    pixel_center.x *= aspect;

    float3 dir = normalize(float3(pixel_center, -1)); // camera origin is 0,0,0
    float minZ = float.maxValue;
    float3 minColor = float3(0, 0, 0);

    for (int i = 0; i < num_shapes; i++)
    {
        Shape shape = shapes[i];
        if (shape.type == ShapeType.Sphere)
        {
            Sphere s = spheres[shape.index];
            float3 xc = s.center;

            float root = ray_sphere_root(s, dir);
            if (root >= minZ || root < 0) continue;

            minZ = root;
            float3 hit = dir * root;
            float3 normal = normalize(hit - xc);
            //float3 tn = (normal + float3(1, 1, 1)) / 2;

            float3 c = float3(0);
            for (int j = 0; j < num_lights; j++)
            {
                c += compute_light(hit, normal, shape.Kd, num_shapes, shapes,
                                   spheres, lights[j]);
            }

            minColor = c;
        }

    }
    if (minZ != float.maxValue)
    {
        image[tid] = float4(minColor, 1.0);
    }
}

float ray_sphere_root(Sphere s, float3 dir, float3 o = float3(0,0,0))
{
    float3 xc = s.center;

    float a = pow(dir.x, 2) + pow(dir.y, 2) + pow(dir.z, 2);
    float b =  2 * ((dir.x * o.x - dir.x * xc.x) + (dir.y * o.y - dir.y * xc.y) + (dir.z * o.z - dir.z * xc.z));
    float c = (pow(o.x, 2) + pow(o.y, 2) + pow(o.z, 2)) - 
              (2 * (xc.x * o.x + xc.y * o.y + xc.z * o.z)) +
              (pow(xc.x, 2) + pow(xc.y, 2) + pow(xc.z, 2)) - pow(s.radius, 2);
    float2 t = float2(-1, -1);
    float root = -1;
    if (solve_quadratic(a, b, c, t))
    {
        // need smallest positive! root
        if (t.y < 0 && t.x >= 0) root = t.x;
        if (t.x < 0 && t.y >= 0) root = t.y;

        if (t.x <= t.y && t.x > 0) root = t.x;
        if (t.y <= t.x && t.y > 0) root = t.y;
    }
    return root;
}

float3 compute_light(float3 p, float3 n, float3 kd,
                     float num_shapes, StructuredBuffer<Shape> shapes,
                     StructuredBuffer<Sphere> spheres, PointLight light)
{
    float3 dir = normalize(light.position - p);
    float tmax = length(light.position - p);
    for (int i = 0; i < num_shapes; i++)
    {
        Shape shape = shapes[i];
        if (shapes[i].type == ShapeType.Sphere)
        {
            Sphere s = spheres[shape.index];
            float root = ray_sphere_root(s, dir, p); // ray origin is not zero in this case
            if (root < EPSILON || root > tmax) continue;

            return float3(0, 0, 0);
        }
    }
    float3 cd = kd * (light.intensity / (pow(tmax, 2))) * max(0, dot(n, dir));
    
    return cd;
}