StructuredBuffer<float3> vertices;
StructuredBuffer<float3> vertex_normals;
StructuredBuffer<float2> vertex_uvs;
uniform float4x4 MVP;
uniform float4x4 model_matrix;
uniform float3x3 normal_matrix; // inverse transpose of the top-left 3x3 above
uniform float3 light_dir;
uniform float3 light_color;
uniform float3 camera_position;
uniform float3 Ks;
uniform float exponent;
Texture2D<float4> Kd_texture;
SamplerState Kd_sampler;

struct VertexOutput {
    float4 position : SV_Position;
    float3 world_position : TEXCOORD0;
    float3 world_normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
};

[shader("vertex")]
VertexOutput vertex_main(uint vid: SV_VertexID) {
    VertexOutput output;
    output.position = mul(MVP, float4(vertices[vid], 1));
    output.world_position = mul(model_matrix, float4(vertices[vid], 1)).xyz;
    output.world_normal = normalize(mul(normal_matrix, vertex_normals[vid]));
    output.uv = vertex_uvs[vid];
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {
    // HW 2.7 TODO: implement
    float3 kd = Kd_texture.Sample(Kd_sampler, input.uv).rgb;
    float3 ca = 0.2 * kd;
    float3 cd = float3(kd.x * light_color.x, kd.y * light_color.y, kd.z * light_color.z) *
                max(0, dot(input.world_normal, light_dir));
    float3 r = 2 * dot(input.world_normal, light_dir) * input.world_normal - light_dir;
    float3 v = normalize(float3(camera_position - input.world_position));
    float3 cs = float3(0, 0, 0);
    if (dot(input.world_normal, light_dir) > 0)
        cs = float3(Ks.x * light_color.x, Ks.y * light_color.y, Ks.z * light_color.z) *
             pow(max(0, dot(r, v)), exponent);
    return float4(ca + cd + cs, 1);
}
