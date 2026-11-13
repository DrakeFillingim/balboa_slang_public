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
    //image[tid] = float4(1, 0, 0, 1);
    float aspect = float(width) / height;
    float2 tid_center = tid + float2(0.5, 0.5);
    // transform from screen to projected camera space

    float new_width = tan(radians(camera.fov_x) / 2);
    float new_height = new_width / aspect;

    float2 pixel_center = 2.0 * (float2(tid_center) / float2(width, height) - float2(0.5, 0.5));
    pixel_center.y *= -new_height;
    pixel_center.x *= new_width;

    /*
    */

    float3 camera_position = xform_pt(camera.cam_to_world, float3(0, 0, 0)); // new camera position in world

    float3 dir = float3(pixel_center, -1);                  // direction of pixel ray // NOTE: not normalized
    float3 world_dir = xform_vec(camera.cam_to_world, dir); // pixel ray in world space
    //image[tid] = float4(1, 0, 0, 1); //NOTE not getting here
    int min_index = compute_ray_intersection(camera_position, world_dir, num_shapes,
                                             shapes, spheres, meshes, vertices, triangles);
    if (min_index < 0) return;
    // code found an intersection
    Shape cur_shape = shapes[min_index];
    //image[tid] = float4(1, 0, 0, 1); //Note: not making it here

    // these values need to be set differently between triangles and spheres
    float3 cur_hit;
    float3 cur_normal;
    if (cur_shape.type == ShapeType.Sphere)
    {
        Sphere s = spheres[cur_shape.index];
        float root = ray_sphere_root(cur_shape, s, world_dir, camera_position);
        cur_hit = camera_position + world_dir * root;
        float3 xc = s.center;
        // find the normal in object space then transform it into world
        cur_normal = normalize(xform_vec(transpose(cur_shape.world_to_obj),
                                                normalize(xform_pt(cur_shape.world_to_obj, cur_hit) - xc)));
    }
    if (cur_shape.type == ShapeType.TriangleMesh)
    {
        //figure out which triangle we intersect
        TriangleMesh mesh = meshes[cur_shape.index];
        uint3 tri = whichTriangle(cur_shape, mesh, camera_position, world_dir, triangles, vertices);
        float3 p0 = xform_pt(cur_shape.obj_to_world, vertices[tri.x]);
        float3 p1 = xform_pt(cur_shape.obj_to_world, vertices[tri.y]);
        float3 p2 = xform_pt(cur_shape.obj_to_world, vertices[tri.z]);
        float minZ = float.maxValue;
        float3 b; //barycentric coords
        float t; //the root
        intersect_tri(p0, p1, p2, camera_position, world_dir, EPSILON, minZ, b, t);
        float3 n0 = vertex_normals[tri.x];
        float3 n1 = vertex_normals[tri.y];
        float3 n2 = vertex_normals[tri.z];
        cur_normal = normalize(xform_vec(transpose(cur_shape.world_to_obj), normalize((b.x) * n0 + b.y * n1 + b.z * n2)));
        cur_hit = camera_position + t * world_dir;
        //image[tid] = float4(1, 0, 0, 1); //NOTE: making it here
    }

    float3 cur_ray = (world_dir);

    float3 c = float3(0);
    float3 throughput = float3(1, 1, 1);

    for (int i = 0; i < max_depth; i++) // for each bounce (if applicable)
    {
        float3 kr = cur_shape.Kr;
        if (kr.x > 0 || kr.y > 0 || kr.z > 0) // is the surface reflective
        {
            c += throughput * compute_contribution(cur_hit, cur_normal, cur_shape,
                                                   num_shapes, shapes, spheres, num_lights, lights, meshes, vertices, triangles);
            throughput *= kr;
            // we shoudl go to object
            cur_ray = normalize(reflect(cur_ray, cur_normal));                                                   // ray reflected from mirror
            int intersectedShapeIndex = compute_ray_intersection(cur_hit, cur_ray, num_shapes, shapes, spheres, meshes, vertices, triangles); // check if reflected ray hits something
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
                if (cur_shape.type == ShapeType.Sphere) {
                    Sphere s = spheres[cur_shape.index];
                    float root = ray_sphere_root(cur_shape, s, cur_ray, cur_hit);
                    cur_hit = cur_hit + root * cur_ray;
                    cur_normal = normalize(xform_vec(transpose(cur_shape.world_to_obj),
                        normalize(xform_pt(cur_shape.world_to_obj, cur_hit) - s.center)));             
                }
                if (cur_shape.type == ShapeType.TriangleMesh) {
                    TriangleMesh mesh = meshes[cur_shape.index];
                    uint3 tri = whichTriangle(cur_shape, mesh, camera_position, world_dir, triangles, vertices);
                    float3 p0 = xform_pt(cur_shape.obj_to_world, vertices[tri.x]);
                    float3 p1 = xform_pt(cur_shape.obj_to_world, vertices[tri.y]);
                    float3 p2 = xform_pt(cur_shape.obj_to_world, vertices[tri.z]);
                    float minZ = float.maxValue;
                    float3 b; //barycentric coords
                    float t; //the root
                    intersect_tri(p0, p1, p2, world_dir, world_dir, EPSILON, minZ, b, t);
                    float3 n0 = vertex_normals[tri.x];
                    float3 n1 = vertex_normals[tri.y];
                    float3 n2 = vertex_normals[tri.z];
                    cur_normal = normalize(xform_vec(transpose(cur_shape.world_to_obj), normalize((b.x * n0 + b.y * n1 + b.z * n2))));
                    cur_hit = t * world_dir;
                }
                             
            }
        }
        else
        {
            // surface has no reflections
            c += throughput * compute_contribution(cur_hit, cur_normal, cur_shape,
                                                   num_shapes, shapes, spheres, num_lights, lights, meshes, vertices, triangles);
            //c = float3(1, 0, 0); //NOTE: not getting here
                                                   break;

        }
    }
    image[tid] = float4(c, 1.0);
}



//returns which triangle you intersected
uint3 whichTriangle(Shape shape, TriangleMesh mesh, float3 o, float3 d, StructuredBuffer<uint3> triangles,
                    StructuredBuffer<float3> vertices) {
    uint3 cur_tri;
    float minZ = float.maxValue;
    uint3 min_tri = uint3(-1, -1, -1);
    for (int i = mesh.triangle_begin; i < mesh.triangle_end; i++) {
        cur_tri = triangles[i] + mesh.vertex_offset;
        float3 p0 = xform_pt(shape.obj_to_world, vertices[cur_tri.x]);
        float3 p1 = xform_pt(shape.obj_to_world, vertices[cur_tri.y]);
        float3 p2 = xform_pt(shape.obj_to_world, vertices[cur_tri.z]);
        float3 b;
        float t;    
        if (intersect_tri(p0, p1, p2, o, d, EPSILON, minZ, b, t))
        {
            if (t < minZ) {
                minZ = t;
                min_tri = cur_tri;
            }
            
        }
    }
    return min_tri;
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

// returns the index of the shape. assumes o and d are in world space.
int compute_ray_intersection(float3 o, float3 d, float num_shapes,
                             StructuredBuffer<Shape> shapes,
                             StructuredBuffer<Sphere> spheres,
                             StructuredBuffer<TriangleMesh> meshes,
                             StructuredBuffer<float3> vertices,
                             StructuredBuffer<uint3> faces)
{
    float minZ = float.maxValue;
    int min_index = -1;
    // we will keep our o and d in world space

    for (int i = 0; i < num_shapes; i++)
    {
        Shape shape = shapes[i];
        if (shape.type == ShapeType.Sphere)
        {
            Sphere s = spheres[shape.index];

            float root = ray_sphere_root(shape, s, d, o);
            if (root >= minZ || root < EPSILON) continue;

            minZ = root;
            min_index = i;
        }
        if (shape.type == ShapeType.TriangleMesh)
        {
            TriangleMesh mesh = meshes[shape.index];
            for (int j = mesh.triangle_begin; j < mesh.triangle_end; j++)
            {
                uint3 tri = faces[j] + mesh.vertex_offset; // check if everything breaks
                float3 p0 = xform_pt(shape.obj_to_world, vertices[tri.x]);
                float3 p1 = xform_pt(shape.obj_to_world, vertices[tri.y]);
                float3 p2 = xform_pt(shape.obj_to_world, vertices[tri.z]);
                float3 b;
                float t;
                if (intersect_tri(p0, p1, p2, o, d, EPSILON, minZ, b, t))
                {
                    minZ = t;
                    min_index = i;
                }
            }
        }
    }
    return min_index;
}

float3 compute_contribution(float3 o, float3 n, Shape s,
                            float num_shapes, StructuredBuffer<Shape> shapes,
                            StructuredBuffer<Sphere> spheres, 
                            float num_lights, StructuredBuffer<PointLight> lights,
                            StructuredBuffer<TriangleMesh> meshes,
                            StructuredBuffer<float3> vertices, StructuredBuffer<uint3> triangles)
{
    float3 c = float3(0);
    for (int i = 0; i < num_lights; i++)
        c += compute_light(o, n, s, num_shapes, shapes, spheres, lights[i], meshes, vertices, triangles);
    return c;
}

float3 compute_light(float3 o, float3 n, Shape s,
                     float num_shapes, StructuredBuffer<Shape> shapes,
                     StructuredBuffer<Sphere> spheres, PointLight light,
                     StructuredBuffer<TriangleMesh> meshes,
                     StructuredBuffer<float3> vertices, StructuredBuffer<uint3> triangles)
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
            float root = ray_sphere_root(shape, sph, dir, o); // ray origin is not zero in this case
            if (root < EPSILON || root > tmax) continue; // light isn't blocked

            return float3(0, 0, 0); // shadow
        }
        if (shape.type == ShapeType.TriangleMesh) {
            //return float3(1, 0, 0); //NOTE: not getting here
            TriangleMesh mesh = meshes[shape.index];
            uint3 tri = whichTriangle(shape, mesh, o, dir, triangles, vertices);
            if (tri.x != -1) { //there was an intersection
                return float3(0, 0, 0);
            }
        }
    }
    // for each shape, is it visible to the camera
    // for this shape, for each light, how is the light affecting the shape
    // for this light, for each shape, is the light getting blocked
    // if the light is not blocked, and if the surface
    // is reflective enough, then trace a ray in the mirror reflection direction
    // and calculate the color based on if you intersect something else

    float3 cd = kd * (light.intensity / (pow(tmax, 2))) * max(0, dot(n, normalize(dir)));

    return cd;
}
