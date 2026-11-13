import hw3_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid: SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
                  uniform int max_depth,
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

    int min_index = compute_ray_intersection(float3(0, 0, 0), dir, num_shapes,
                                             shapes, spheres);
    if (min_index < 0) return;

    Shape cur_shape = shapes[min_index];

    Sphere s = spheres[cur_shape.index];
    float root = ray_sphere_root(spheres[cur_shape.index], dir);
    float3 hit = dir * root;
    float3 xc = s.center;

    float3 cur_hit = hit;
    float3 cur_ray = dir;
    float3 cur_normal = normalize(hit - xc);

    float3 c = float3(0);
    float3 throughput = float3(1, 1, 1);

    for (int i = 0; i < max_depth; i++) // for each bounce (if applicable)
    {
        float3 kr = cur_shape.Kr;
        if (kr.x > 0 || kr.y > 0 || kr.z > 0) // is the surface reflective
        {
                c += throughput * compute_contribution(cur_hit, cur_normal, cur_shape,
                    num_shapes, shapes, spheres, num_lights, lights);
                throughput *= kr;

                cur_ray = normalize(reflect(cur_ray, cur_normal)); // ray reflected from mirror
                int intersectedShapeIndex = compute_ray_intersection(cur_hit, cur_ray, num_shapes, shapes, spheres); //check if reflected ray hits something
                if (intersectedShapeIndex == -1)
                {
                    // ray hits nothing, hits background
                    c += throughput * background;
                    break;
                }
                else
                {
                    // reflected ray hits something else
                    // current shape is now intersected shape
                    cur_shape = shapes[intersectedShapeIndex];
                    Sphere s = spheres[cur_shape.index];
                    float root = ray_sphere_root(s, cur_ray, cur_hit);
                    cur_hit = cur_hit + root * cur_ray;
                    cur_normal = normalize(cur_hit - s.center);
                    
                }
            }
            else
            {
                // surface has no reflections
                c += throughput * compute_contribution(cur_hit, cur_normal, cur_shape,
                                          num_shapes, shapes, spheres, num_lights, lights);
                break;
            }
    }
    image[tid] = float4(c, 1.0);
}

float ray_sphere_root(Sphere s, float3 dir, float3 o = float3(0, 0, 0))
{
    float3 xc = s.center;

    float a = pow(dir.x, 2) + pow(dir.y, 2) + pow(dir.z, 2);
    float b = 2 * ((dir.x * o.x - dir.x * xc.x) + (dir.y * o.y - dir.y * xc.y) + (dir.z * o.z - dir.z * xc.z));
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

// returns the index of the shape.
int compute_ray_intersection(float3 o, float3 d, float num_shapes,
                             StructuredBuffer<Shape> shapes,
                             StructuredBuffer<Sphere> spheres)
{
    float minZ = float.maxValue;
    int min_index = -1;
    for (int i = 0; i < num_shapes; i++)
    {
        Shape shape = shapes[i];
        if (shape.type == ShapeType.Sphere)
        {
            Sphere s = spheres[shape.index];
            float3 xc = s.center;

            float root = ray_sphere_root(s, d, o);
            if (root >= minZ || root < EPSILON) continue;

            minZ = root;
            min_index = i;            
        }
    }
    return min_index;
}

float3 compute_contribution(float3 o, float3 n, Shape s,
                            float num_shapes, StructuredBuffer<Shape> shapes,
                            StructuredBuffer<Sphere> spheres, float num_lights,
                            StructuredBuffer<PointLight> lights)
{
    float3 c = float3(0);
    for (int i = 0; i < num_lights; i++)
        c += compute_light(o, n, s, num_shapes, shapes, spheres, lights[i]);
    return c;
}

float3 compute_light(float3 o, float3 n, Shape s,
                     float num_shapes, StructuredBuffer<Shape> shapes,
                     StructuredBuffer<Sphere> spheres, PointLight light)
{
    float3 kd = s.Kd;
    float3 dir = normalize(light.position - o);
    float tmax = length(light.position - o);
    for (int i = 0; i < num_shapes; i++)
    {
        Shape shape = shapes[i];
        if (shapes[i].type == ShapeType.Sphere) 
        {
            Sphere sph = spheres[shape.index];
            float root = ray_sphere_root(sph, dir, o); // ray origin is not zero in this case
            if (root < EPSILON || root > tmax) continue; //light isn't blocked

            return float3(0, 0, 0); //shadow
        } 
    }
    // for each shape, is it visible to the camera
    // for this shape, for each light, how is the light affecting the shape
    // for this light, for each shape, is the light getting blocked
    // if the light is not blocked, and if the surface
    // is reflective enough, then trace a ray in the mirror reflection direction
    // and calculate the color based on if you intersect something else
    
    float3 cd = kd * (light.intensity / (pow(tmax, 2))) * max(0, dot(n, dir));

    return cd;
}