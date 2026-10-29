[shader("compute")]
[numthreads(64, 1, 1)]
void clear_buffers(uint tid : SV_DispatchThreadID,
                   uniform uint num_pixels,
                   RWStructuredBuffer<uint> depth_buffer,
                   RWStructuredBuffer<uint> triangle_id_buffer) {
    if (tid >= num_pixels)
        return;

    depth_buffer[tid] = 0xffffffff;
    triangle_id_buffer[tid] = 0xffffffff;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void transform_vertices(
    uint tid : SV_DispatchThreadID,
    uniform uint num_vertices,
    uniform uint vertex_offset,
    uniform float4x4 MVP,
    StructuredBuffer<float3> vertices,
    RWStructuredBuffer<float3> projected_vertices)
{
    if (tid >= num_vertices)
        return;

    uint i = vertex_offset + tid;
    float4 p = mul(MVP, float4(vertices[i], 1.0));
    projected_vertices[i] = p.xyz / p.w;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_depth(
        uint tid : SV_DispatchThreadID,
        uniform uint num_triangles,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        // A projected vertex store the screen space coordinates
        // in xy, and the inverse depth in z.
        StructuredBuffer<float3> projected_vertices,
        RWStructuredBuffer<uint> depth_buffer) {
    // HW 2.5 TODO

    if (tid >= num_triangles) return;
    uint3 f = faces[tid];
    float2 p0 = projected_vertices[f.x].xy;
    float z0 = projected_vertices[f.x].z;
    float2 p1 = projected_vertices[f.y].xy;
    float z1 = projected_vertices[f.y].z;
    float2 p2 = projected_vertices[f.z].xy;
    float z2 = projected_vertices[f.z].z;

    float2 aabb_min = float2(floor(min(min(p0.x, p1.x), p2.x)),
                             floor(min(min(p0.y, p1.y), p2.y)));
    float2 aabb_max = float2(ceil(max(max(p0.x, p1.x), p2.x)),
                             ceil(max(max(p0.y, p1.y), p2.y)));

    for (int y = aabb_min.y; y < aabb_max.y; y++)
    {
        for (int x = aabb_min.x; x < aabb_max.x; x++)
        {
            float2 pixel_pos = float2(x + 0.5, y + 0.5);
            if (in_triangle(pixel_pos, p0, p1, p2))
            {
                float3 w = compute_barycentric_weights(pixel_pos, p0, p1, p2, z0, z1, z2);
                if (w.x < 0) continue;
                float d = w.x / z0 + w.y / z1 + w.z / z2;
                uint d_bits = asuint(d);
                InterlockedMin(depth_buffer[y * width + x], d_bits);
            }
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_id(
        uint tid : SV_DispatchThreadID,
        uniform uint num_triangles,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        StructuredBuffer<float3> projected_vertices,
        RWStructuredBuffer<uint> depth_buffer,
        RWStructuredBuffer<uint> triangle_id_buffer) {
    // HW 2.5 TODO

    if (tid >= num_triangles) return;
    uint3 f = faces[tid];
    float2 p0 = projected_vertices[f.x].xy;
    float z0 = projected_vertices[f.x].z;
    float2 p1 = projected_vertices[f.y].xy;
    float z1 = projected_vertices[f.y].z;
    float2 p2 = projected_vertices[f.z].xy;
    float z2 = projected_vertices[f.z].z;

    uint2 aabb_min = uint2(floor(min(min(p0.x, p1.x), p2.x)),
                             floor(min(min(p0.y, p1.y), p2.y)));
    uint2 aabb_max = uint2(ceil(max(max(p0.x, p1.x), p2.x)),
                             ceil(max(max(p0.y, p1.y), p2.y)));

    for (int y = aabb_min.y; y < aabb_max.y; y++)
    {
        for (int x = aabb_min.x; x < aabb_max.x; x++)
        {
            float2 pixel_pos = float2(x + 0.5, y + 0.5);
            if (in_triangle(pixel_pos, p0, p1, p2))
            {
                float3 w = compute_barycentric_weights(pixel_pos, p0, p1, p2, z0, z1, z2);
                if (w.x < 0) continue;
                float d = w.x / z0 + w.y / z1 + w.z / z2;
                if (asuint(d) == depth_buffer[y * width + x]) 
                    InterlockedMin(triangle_id_buffer[y * width + x], tid);
            }
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void shade(
        uint2 tid : SV_DispatchThreadID,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        StructuredBuffer<float3> vertex_colors,
        StructuredBuffer<float3> projected_vertices,
        StructuredBuffer<uint> triangle_id_buffer,
        RWTexture2D<float4> image) {
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    float2 pixel_pos = tid + float2(0.5, 0.5);
    // HW 2.5 TODO
    uint face_id = triangle_id_buffer[tid.y * width + tid.x];
    if (face_id != 0xffffffff)
    {
        uint3 f = faces[face_id];
        float2 p0 = projected_vertices[f.x].xy;
        float z0 = projected_vertices[f.x].z;
        float2 p1 = projected_vertices[f.y].xy;
        float z1 = projected_vertices[f.y].z;
        float2 p2 = projected_vertices[f.z].xy;
        float z2 = projected_vertices[f.z].z;

        float3 c0 = vertex_colors[f.x];
        float3 c1 = vertex_colors[f.y];
        float3 c2 = vertex_colors[f.z];

        float3 w = compute_barycentric_weights(pixel_pos, p0, p1, p2, z0, z1, z2);
        if (w.x >= 0) image[tid] = float4(w.x * c0 + w.y * c1 + w.z * c2, 1);
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

float3 compute_barycentric_weights(float2 pixel_pos, float2 sp0, float2 sp1, float2 sp2,
                          float z0, float z1, float z2)
{
    float totalArea = length(cross(float3(sp1 - sp0, 0), float3(sp2 - sp0, 0)));

    if (totalArea <= 0) return float3(-1, -1, -1);

    float b0p = length(cross(float3(sp1 - pixel_pos, 0), float3(sp2 - pixel_pos, 0))) / totalArea;
    float b1p = length(cross(float3(sp0 - pixel_pos, 0), float3(sp2 - pixel_pos, 0))) / totalArea;
    float b2p = length(cross(float3(sp0 - pixel_pos, 0), float3(sp1 - pixel_pos, 0))) / totalArea;

    // finding b0, b1, b2
    float denom = b0p * z0 + b1p * z1 + b2p * z2;
    float b0 = (b0p * z0) / denom;
    float b1 = (b1p * z1) / denom;
    float b2 = (b2p * z2) / denom;

    return float3(b0, b1, b2);
    //return b0 / z0 + b1 / z1 + b2 / z2;
}