import hw3_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Sphere> spheres,
                  uniform int num_spheres,
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
    float3 minNorm = float3(0, 0, 0);

    for (int i = 0; i < num_spheres; i++)
    {
        Sphere s = spheres[i];
        float3 xc = s.center;

        float a = pow(dir.x, 2) + pow(dir.y, 2) + pow(dir.z, 2);
        float b = -2 * ((dir.x * xc.x) + (dir.y * xc.y) + (dir.z * xc.z));
        float c = pow(xc.x, 2) + pow(xc.y, 2) + pow(xc.z, 2) - pow(s.radius, 2);
        float2 t = float2(-1, -1);
        if (solve_quadratic(a, b, c, t))
        {
            // need smallest positive! root
            float root = -1;
            if (t.y < 0 && t.x >= 0) root = t.x;
            if (t.x < 0 && t.y >= 0) root = t.y;

            if (t.x <= t.y && t.x > 0) root = t.x;
            if (t.y <= t.x && t.y > 0) root = t.y;

            if (root >= minZ || root < 0) continue;

            minZ = root;
            float3 normal = normalize(dir * root - xc);
            float3 tn = (normal + float3(1, 1, 1)) / 2;
            minNorm = tn;
        }
    }
    if (minZ != float.maxValue)
    {
        image[tid] = float4(minNorm, 1.0);
    }

}
