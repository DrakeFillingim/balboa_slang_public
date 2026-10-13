[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 center,
                  uniform float radius,
                  uniform float3 color,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    uint2 outPos = uint2(tid.x, height - 1 - tid.y);

    image[outPos] = float4(1, 1, 1, 1);
    // TODO: your code here

    float l = length(center - (tid + float2(0.5, 0.5)));
    if (l < radius) image[outPos] = float4(color, 1);
}
