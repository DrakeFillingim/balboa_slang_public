[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0,
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform int2 aabb_min,
                  uniform float3 color,
                  RWTexture2D<float4> image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    // TODO: HW 2.1

    //image[ipixel] = float4(float3(0.0), 1.0);

    float2 pixel_pos = ipixel + float2(0.5, 0.5);
    if (in_triangle(pixel_pos, sp0, sp1, sp2)) image[ipixel] = float4(color, 1);
}

bool in_triangle(float2 q, float2 p0, float2 p1, float2 p2)
{
    float2 e01 = p1 - p0;
    float2 e12 = p2 - p1;
    float2 e20 = p0 - p2;
    float2 n01 = float2(e01.y, -e01.x);
    float2 n12 = float2(e12.y, -e12.x);
    float2 n20 = float2(e20.y, -e20.x);
    float d0 = dot(q - p0, n01);
    float d1 = dot(q - p1, n12);
    float d2 = dot(q - p2, n20);
    return (d0 > 0 && d1 > 0 && d2 > 0) || (d0 < 0 && d1 < 0 && d2 < 0);
}