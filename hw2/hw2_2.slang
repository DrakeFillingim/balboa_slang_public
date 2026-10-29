[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0, // vertex at screen space
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform float inv_depth0, // inverse depth in camera space
                  uniform float inv_depth1,
                  uniform float inv_depth2,
                  uniform int2 aabb_min,
                  uniform float3 c0, // vertex color
                  uniform float3 c1,
                  uniform float3 c2,
                  uniform float s,
                  uniform float a,
                  RWTexture2D<float4> color_image,
                  RWTexture2D<float> depth_image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    color_image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    // TODO: HW 2.1

    float2 pixel_pos = ipixel + float2(0.5, 0.5);

    /*if (inv_depth0 < 0) return;
    if (inv_depth1 < 0) return;
    if (inv_depth2 < 0) return;*/

    // p, p0, p1, p2 are the real things
    // p', p1', p2', p3' are the screen space versions

    // depths/unprojected z's
    float z0 = inv_depth0;
    float z1 = inv_depth1;
    float z2 = inv_depth2;

    // b0', b1', b2', cross product of two edge vectors
    float totalArea = length(cross(float3(sp1 - sp0, 0), float3(sp2 - sp0, 0)));
    float b0p = length(cross(float3(sp1 - pixel_pos, 0), float3(sp2 - pixel_pos, 0))) / totalArea;
    float b1p = length(cross(float3(sp0 - pixel_pos, 0), float3(sp2 - pixel_pos, 0))) / totalArea;
    float b2p = length(cross(float3(sp0 - pixel_pos, 0), float3(sp1 - pixel_pos, 0))) / totalArea;

    // finding b0, b1, b2
    float denom = b0p * z0 + b1p * z1 + b2p * z2;
    float b0 = (b0p * z0) / denom;
    float b1 = (b1p * z1) / denom;
    float b2 = (b2p * z2) / denom;

    float pz = b0 / z0 + b1 / z1 + b2 / z2;

    if (pz < depth_image[ipixel] && in_triangle(pixel_pos, sp0, sp1, sp2))
    {
        depth_image[ipixel] = pz;
        float3 c = b0 * c0 + b1 * c1 + b2 * c2;
        color_image[ipixel] = float4(c, 1.0);
    }
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
    return (d0 >= 0 && d1 >= 0 && d2 >= 0) || (d0 <= 0 && d1 <= 0 && d2 <= 0);
}
