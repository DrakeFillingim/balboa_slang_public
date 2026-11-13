import hw3_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid: SV_DispatchThreadID,
                  uniform float3 background,
                  uniform Camera camera,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<TriangleMesh> meshes,
                  StructuredBuffer<float3> vertices,
                  StructuredBuffer<float3> vertex_normals,
                  StructuredBuffer<uint3> triangles,
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
    // transform from screen to projected camera space

    float new_width = tan(radians(camera.fov_x) / 2);
    float new_height = new_width / aspect;

    float2 pixel_center = 2.0 * (float2(tid_center) / float2(width, height) - float2(0.5, 0.5));
    pixel_center.y *= -new_height;
    pixel_center.x *= new_width;

    float3 camera_position = xform_pt(camera.cam_to_world, float3(0, 0, 0)); // new camera position in world

    float3 dir = float3(pixel_center, -1);                  // direction of pixel ray // NOTE: not normalized
    float3 world_dir = xform_vec(camera.cam_to_world, dir); // pixel ray in world space

    Intersection intsec = compute_ray_intersection(camera_position, world_dir, num_shapes,
                                             shapes, spheres, meshes, vertices, triangles);
    if (!intsec.found) return;

    // found an intersection
    Shape cur_shape = intsec.shape;
    float3 cur_hit = camera_position + intsec.root * world_dir;
    float3 cur_normal = compute_normal(intsec, cur_hit, vertex_normals);
    float3 cur_ray = world_dir;

    float3 c = float3(0);
    float3 throughput = float3(1, 1, 1);

    for (int i = 0; i < max_depth; i++) // for each bounce (if applicable)
    {
        float3 kr = cur_shape.Kr;
        if (kr.x > 0 || kr.y > 0 || kr.z > 0) // is the surface reflective
        {
            c += throughput * compute_contribution(cur_hit, cur_normal, cur_shape,
                                                   num_shapes, shapes, spheres, meshes,
                                                   vertices, triangles, num_lights, lights);
            throughput *= kr;
            // we shoudl go to object
            cur_ray = normalize(reflect(cur_ray, cur_normal));                                                   // ray reflected from mirror
            Intersection new_intsec = compute_ray_intersection(cur_hit, cur_ray, num_shapes, shapes, spheres,
                                                               meshes, vertices, triangles); // check if reflected ray hits something
            if (!new_intsec.found)
            {
                // ray hits nothing, hits background
                c += throughput * background;
                break;
            }
            else
            {
                // reflected ray hits something else
                // current shape is now intersected shape
                cur_shape = new_intsec.shape;
                cur_hit = cur_hit + new_intsec.root * cur_ray;
                cur_normal = compute_normal(new_intsec, cur_hit, vertex_normals);
            }
        }
        else
        {
            // surface has no reflections
            c += throughput * compute_contribution(cur_hit, cur_normal, cur_shape,
                                                   num_shapes, shapes, spheres, meshes, vertices,
                                                   triangles, num_lights, lights);
            break;
        }
    }
    image[tid] = float4(c, 1.0);
}

// normal needs to be set differently between triangles and spheres
// find the normal in object space then transform it into world
float3 compute_normal(Intersection intsec, float3 hit, StructuredBuffer<float3> vertex_normals)
{
    Shape s = intsec.shape;
    float3 n;
    if (s.type == ShapeType.Sphere)
        n = normalize(xform_vec(transpose(s.world_to_obj),
                                normalize(xform_pt(s.world_to_obj, hit) - intsec.sphere.center)));

    if (intsec.shape.type == ShapeType.TriangleMesh)
    {
        float b0 = intsec.barycentric.x;
        float b1 = intsec.barycentric.y;
        float b2 = intsec.barycentric.z;
        float3 n0 = vertex_normals[intsec.tri.x];
        float3 n1 = vertex_normals[intsec.tri.y];
        float3 n2 = vertex_normals[intsec.tri.z];
        n = normalize(xform_vec(transpose(s.world_to_obj), normalize(b0 * n0 + b1 * n1 + b2 * n2)));
    }
    return n;
}

// assumes dir and o are in world space.
float ray_sphere_root(Shape shape, Sphere s, float3 dir, float3 o = float3(0, 0, 0))
{
    dir = xform_vec(shape.world_to_obj, dir);
    o = xform_pt(shape.world_to_obj, o);
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

// creates an intersection struct for the closest shape.
// assumes o (origin) and d (direction) are in world space.
Intersection compute_ray_intersection(float3 o, float3 d, float num_shapes,
                             StructuredBuffer<Shape> shapes,
                             StructuredBuffer<Sphere> spheres,
                             StructuredBuffer<TriangleMesh> meshes,
                             StructuredBuffer<float3> vertices,
                             StructuredBuffer<uint3> faces)
{
    float minZ = float.maxValue;
    Intersection intersection = { false };

    for (int i = 0; i < num_shapes; i++)
    {
        Shape shape = shapes[i];
        if (shape.type == ShapeType.Sphere)
        {
            Sphere s = spheres[shape.index];

            float root = ray_sphere_root(shape, s, d, o);
            if (root >= minZ || root < EPSILON) continue;

            minZ = root;
            intersection = {
                true,
                shape,
                root,
                s
            };
        }
        if (shape.type == ShapeType.TriangleMesh)
        {
            TriangleMesh mesh = meshes[shape.index];
            for (int j = mesh.triangle_begin; j < mesh.triangle_end; j++)
            {
                uint3 tri = faces[j] + mesh.vertex_offset;
                float3 p0 = xform_pt(shape.obj_to_world, vertices[tri.x]);
                float3 p1 = xform_pt(shape.obj_to_world, vertices[tri.y]);
                float3 p2 = xform_pt(shape.obj_to_world, vertices[tri.z]);
                float3 b;
                float t;
                if (intersect_tri(p0, p1, p2, o, d, EPSILON, minZ, b, t) && t < minZ)
                {
                    minZ = t;
                    intersection = {
                        true,
                        shape,
                        t,
                        mesh,
                        tri,
                        b
                    };
                }
            }
        }
    }
    return intersection;
}

float3 compute_contribution(float3 o, float3 n, Shape s,
                            float num_shapes, StructuredBuffer<Shape> shapes,
                            StructuredBuffer<Sphere> spheres, StructuredBuffer<TriangleMesh> meshes,
                            StructuredBuffer<float3> vertices, StructuredBuffer<uint3> faces,
                            float num_lights, StructuredBuffer<PointLight> lights)
{
    float3 c = float3(0);
    for (int i = 0; i < num_lights; i++)
        c += compute_light(o, n, s, num_shapes, shapes, spheres, meshes, vertices, faces, lights[i]);
    return c;
}

float3 compute_light(float3 o, float3 n, Shape s,
                     float num_shapes, StructuredBuffer<Shape> shapes,
                     StructuredBuffer<Sphere> spheres, StructuredBuffer<TriangleMesh> meshes,
                     StructuredBuffer<float3> vertices, StructuredBuffer<uint3> faces, PointLight light)
{
    float3 kd = s.Kd;
    float3 dir = normalize(light.position - o);
    float tmax = length(light.position - o);
    Intersection intsec = compute_ray_intersection(o, dir, num_shapes, shapes, spheres,
                                                   meshes, vertices, faces);
    /* for each shape, is it visible to the camera
     for this shape, for each light, how is the light affecting the shape
     for this light, for each shape, is the light getting blocked
     if the light is not blocked, and if the surface
     is reflective enough, then trace a ray in the mirror reflection direction
     and calculate the color based on if you intersect something else
    */

    if (intsec.found && intsec.root < tmax && intsec.root > EPSILON) return float3(0, 0, 0);

    float3 cd = kd * (light.intensity / (pow(tmax, 2))) * max(0, dot(n, normalize(dir)));    
    return cd;
}

struct Intersection
{
    bool found;
    Shape shape;
    float root;
    Sphere sphere;
    TriangleMesh mesh;
    uint3 tri;
    float3 barycentric;

    __init(bool _found)
    {
        found = _found;
    }

    __init(bool _found, Shape _shape, float _root, Sphere _sphere)
    {
        found = _found;
        shape = _shape;
        root = _root;
        sphere = _sphere;
    }

    __init(bool _found, Shape _shape, float _root, TriangleMesh _mesh, uint3 _tri, float3 _barycentric)
    {
        found = _found;
        shape = _shape;
        root = _root;
        mesh = _mesh;
        tri = _tri;
        barycentric = _barycentric;
    }
}