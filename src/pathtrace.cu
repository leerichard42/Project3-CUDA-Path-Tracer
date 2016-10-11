#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that processes the current image data based on number of samples and antialiasing.
__global__ void processImage(glm::ivec2 resolution,
	int iter, glm::vec3* image_data, glm::vec3* image_out, int antiAliasing, glm::vec3* image_edges) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		bool supersample = antiAliasing == ADAPTIVE && image_edges[index].x > 0;
		glm::vec3 pix = image_data[index];

		glm::vec3 color;
		if (antiAliasing == NOAA || (antiAliasing == ADAPTIVE && !supersample)) {
			color.x = glm::clamp(pix.x / iter, 0.f, 1.f);
			color.y = glm::clamp(pix.y / iter, 0.f, 1.f);
			color.z = glm::clamp(pix.z / iter, 0.f, 1.f);
		}
		else if (antiAliasing == AA || (antiAliasing == ADAPTIVE && supersample)) {
			color.x = glm::clamp(pix.x / (4.0f * iter), 0.f, 1.f);
			color.y = glm::clamp(pix.y / (4.0f * iter), 0.f, 1.f);
			color.z = glm::clamp(pix.z / (4.0f * iter), 0.f, 1.f);
		}

		image_out[index] = color;
	}
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, int pixelCount, glm::vec3* image) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < pixelCount) {
		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = (int)(image[index].x * 255);
		pbo[index].y = (int)(image[index].y * 255);
		pbo[index].z = (int)(image[index].z * 255);
	}
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static glm::vec3 * dev_image_normals = NULL;
static glm::vec3 * dev_image_edges = NULL;
static glm::vec3 * dev_image_result = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static PathSegment * dev_paths_cached = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static ShadeableIntersection * dev_intersections_cached = NULL;
thrust::device_ptr<PathSegment> dev_thrust_paths;
thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections;
static int antiAliasing = 0;
static int start_paths = 0;

void pathtraceInit(Scene *scene, int aa_state) {
	hst_scene = scene;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_image_edges, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image_edges, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_image_normals, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image_normals, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_image_result, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image_result, 0, pixelcount * sizeof(glm::vec3));

	antiAliasing = aa_state;
	start_paths = pixelcount;
	if (antiAliasing == AA || antiAliasing == ADAPTIVE){
		start_paths *= 4;
	}

	cudaMalloc(&dev_paths, start_paths * sizeof(PathSegment));
	cudaMalloc(&dev_paths_cached, start_paths * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, start_paths * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, start_paths * sizeof(ShadeableIntersection));
	cudaMalloc(&dev_intersections_cached, start_paths * sizeof(ShadeableIntersection));

	dev_thrust_paths = thrust::device_pointer_cast(dev_paths);
	dev_thrust_intersections = thrust::device_pointer_cast(dev_intersections);

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	cudaFree(dev_image_edges);
	cudaFree(dev_image_normals);
	cudaFree(dev_image_result);
	cudaFree(dev_paths);
	cudaFree(dev_paths_cached);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	cudaFree(dev_intersections_cached);

	checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, int antiAliasing, glm::vec3* image_edges)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		thrust::uniform_real_distribution<float> u_jitter(-0.1, 0.1);
		bool supersample = antiAliasing == ADAPTIVE && image_edges[index].x > 0;

		if (antiAliasing == NOAA) {
			PathSegment & segment = pathSegments[index];

			segment.ray.origin = cam.position;
			segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
			segment.isTerminated = false;

			segment.ray.direction = glm::normalize(cam.view
				- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
				- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
				);

			segment.pixelIndex = index;
			segment.remainingBounces = traceDepth;
			segment.pathId = index;
		}
		else if (antiAliasing == AA || antiAliasing == ADAPTIVE) {
			for (int i = 0; i < 4; i++) {
				thrust::default_random_engine rng_x = makeSeededRandomEngine(0, index, i);
				thrust::default_random_engine rng_y = makeSeededRandomEngine(0, index, 2 * i);

				PathSegment & segment = pathSegments[4 * index + i];

				segment.ray.origin = cam.position;
				segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
				segment.isTerminated = false;

				if (antiAliasing == AA || supersample) {
					segment.ray.direction = glm::normalize(cam.view
						- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
						+ cam.right * cam.pixelLength.x * ((float)(i % 2) * 0.5f - 0.25f + u_jitter(rng_x))
						- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
						+ cam.up * cam.pixelLength.y * ((float)(i / 2) * 0.5f - 0.25f + u_jitter(rng_y))
						);
				}
				else {
					if (i == 0) {
						segment.ray.direction = glm::normalize(cam.view
							- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
							- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
							);
					}
					else {
						segment.isTerminated = true;
					}
				}

				segment.pixelIndex = index;
				segment.remainingBounces = traceDepth;
				segment.pathId = 4 * index + i;
			}
		}
	}
}

//Generate buffer with normal information
__global__ void generateNormalData(Camera cam, int pixelCount, glm::vec3* image, Material * materials, ShadeableIntersection * intersections)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < pixelCount) {
		if (materials[intersections[index].materialId].emittance) {
			image[index] = glm::vec3(1.0f);
		}
		else {
			image[index] = (0.5f * intersections[index].surfaceNormal + glm::vec3(0.5f)) *
				materials[intersections[index].materialId].color;
		}
	}
}

__device__ float threshold(float min, float max, float val) {
	if (val < min) { return 0.0f; }
	if (val > max) { return 1.0f; }
	return val;
}

//Generate edge buffer from normal pass
__global__ void generateEdgeData(Camera cam, glm::vec3* image_normals, glm::vec3* image_edges)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x > 0 && x < cam.resolution.x &&
		y > 0 && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		float pix[9];
		int k = -1;

		// Read surrounding pixels
		for (int i = -1; i < 2; i++) {
			for (int j = -1; j < 2; j++) {
				k++;
				glm::vec3 color = image_normals[(x + i) + ((y + j) * cam.resolution.x)];
				pix[k] = (color.r + color.g + color.b) / 3.0f;
			}
		}

		// Average color differences
		float delta = (abs(pix[1] - pix[7]) +
			abs(pix[5] - pix[3]) +
			abs(pix[0] - pix[8]) +
			abs(pix[2] - pix[6])) / 4.0f;

		if (glm::clamp(10.0f*delta, 0.0f, 1.0f) > 0.3f){
			image_edges[index] = glm::vec3(1.0f);
		}
	}
}

__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
	, Material * materials
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1 || t_min < 0.005f)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;

			if (dot(normal, pathSegment.ray.direction) > 0){
				intersections[path_index].entering = false;
				if (materials[geoms[hit_geom_index].materialid].hasRefractive){
					normal *= -1;
				}
			}
			else{
				intersections[path_index].entering = true;
			}

			intersections[path_index].surfaceNormal = normal;
		}
	}
}

__global__ void shadeMaterial(
	int depth
	, int traceDepth
	, int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments_in
	, PathSegment * pathSegments_out
	, Material * materials
	)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths/* && !pathSegments_in[idx].isTerminated*/)
	{
		if (depth == 0) {
			pathSegments_out[idx].color = glm::vec3(1.0f, 1.0f, 1.0f);
			pathSegments_out[idx].isTerminated = pathSegments_in[idx].isTerminated;

			pathSegments_out[idx].pixelIndex = pathSegments_in[idx].pixelIndex;
			pathSegments_out[idx].remainingBounces = traceDepth;
			pathSegments_out[idx].pathId = pathSegments_in[idx].pathId;
		}

		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
			// Set up the RNG
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, pathSegments_in[idx].pixelIndex, pathSegments_in[idx].remainingBounces);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments_out[idx].color *= (materialColor * material.emittance);
				pathSegments_out[idx].isTerminated = true;
			}
			else {
				//Perfect Specular
				if (material.hasReflective) {
					pathSegments_out[idx].ray.origin = pathSegments_in[idx].ray.origin + intersection.t * pathSegments_in[idx].ray.direction + intersection.surfaceNormal * 0.001f;
					pathSegments_out[idx].ray.direction = reflect(pathSegments_in[idx].ray.direction, intersection.surfaceNormal);
					pathSegments_out[idx].color *= material.specular.color;
				}
				else if (material.hasRefractive) {
					if (material.indexOfRefraction < FLT_EPSILON){
						printf("Check index of refraction!\n");
					}
					float w = powf(1.0f - abs(dot(intersection.surfaceNormal, pathSegments_in[idx].ray.direction)), 5.0f);
					float r0 = powf((1.0f - material.indexOfRefraction) / (1.0f + material.indexOfRefraction), 2.0f);
					float fresnel = r0 + (1 - r0) * w;

					float eta;
					if (intersection.entering){
						eta = 1.f / material.indexOfRefraction;
					}
					else{
						eta = material.indexOfRefraction / 1.f;
					}

					if (intersection.entering && u01(rng) < fresnel){
						pathSegments_out[idx].ray.origin = pathSegments_in[idx].ray.origin + intersection.t * pathSegments_in[idx].ray.direction + intersection.surfaceNormal * 0.001f;
						pathSegments_out[idx].ray.direction = reflect(pathSegments_in[idx].ray.direction, intersection.surfaceNormal);
					}
					else {
						pathSegments_out[idx].ray.origin = pathSegments_in[idx].ray.origin + intersection.t * pathSegments_in[idx].ray.direction - intersection.surfaceNormal * 0.001f;
						pathSegments_out[idx].ray.direction = refract(normalize(pathSegments_in[idx].ray.direction), normalize(intersection.surfaceNormal), eta);
					}
					pathSegments_out[idx].color *= materialColor;
				}
				//Imperfect Specular
				else if (material.specular.exponent > 0) {
				}
				//Diffuse
				else {
					pathSegments_out[idx].ray.origin = pathSegments_in[idx].ray.origin + intersection.t * pathSegments_in[idx].ray.direction + intersection.surfaceNormal * 0.001f;
					pathSegments_out[idx].ray.direction = calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng);
					pathSegments_out[idx].color *= materialColor;
				}

				pathSegments_out[idx].remainingBounces = pathSegments_in[idx].remainingBounces - 1;
				if (pathSegments_out[idx].remainingBounces == 0) {
					pathSegments_out[idx].color = glm::vec3(0.0f);
					pathSegments_out[idx].isTerminated = true;
				}
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else {
			pathSegments_out[idx].color = glm::vec3(0.0f);
			pathSegments_out[idx].isTerminated = true;
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(glm::ivec2 resolution, glm::vec3 * image, PathSegment * iterationPaths, int antiAliasing, glm::vec3* image_edges)
{
	//int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		bool supersample = antiAliasing == ADAPTIVE && image_edges[index].x > 0;

		if (antiAliasing == AA || (antiAliasing == ADAPTIVE && supersample)) {
			image[index] += iterationPaths[4 * index].color + 
				iterationPaths[4 * index + 1].color + 
				iterationPaths[4 * index + 2].color + 
				iterationPaths[4 * index + 3].color;
		}
		else {
			PathSegment iterationPath = iterationPaths[antiAliasing == ADAPTIVE ? 4 * index : index];
			image[iterationPath.pixelIndex] += iterationPath.color;
		}
	}
}

struct is_not_terminated
{
	__host__ __device__ bool operator()(const PathSegment& path)
	{
		return !path.isTerminated;
	}
};

struct material_id_comparator
{
	__host__ __device__ bool operator()(const ShadeableIntersection& i1, const ShadeableIntersection& i2)
	{
		return i1.materialId < i2.materialId;
	}
};

struct path_id_comparator
{
	__host__ __device__ bool operator()(const PathSegment& p1, const PathSegment& p2)
	{
		return p1.pathId < p2.pathId;
	}
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter, bool sortByMat, bool cacheFirstBounce, bool viewEdges) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;

	int depth = 0;
	int num_paths = start_paths;

	//Generate buffer with edge information
	if (antiAliasing == ADAPTIVE && (iter == 1)) {
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths, NOAA, dev_image_edges);

		computeIntersections << <numBlocksPixels, blockSize1d >> > (
			depth, pixelcount, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections, dev_materials);

		generateNormalData << <numBlocksPixels, blockSize1d >> > (cam, pixelcount, dev_image_normals, dev_materials, dev_intersections);

		generateEdgeData << <blocksPerGrid2d, blockSize2d >> > (cam, dev_image_normals, dev_image_edges);
	}

	// Raycast
	if (!cacheFirstBounce || (iter == 1)) {
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> >(cam, iter, traceDepth, dev_paths, antiAliasing, dev_image_edges);
		checkCUDAError("generate camera ray");
	}

	if (iter == 1) {
		cudaMemcpy(dev_paths_cached, dev_paths, sizeof(PathSegment) * start_paths, cudaMemcpyDeviceToDevice);
	}

	// Shoot rays into scene, bounce between objects, push shading chunks
	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));

		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		// Intersection testing
		if ((depth == 0 && iter == 1) || !cacheFirstBounce || (cacheFirstBounce && depth > 0)) {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections, dev_materials);
			checkCUDAError("computeIntersections failed!");
			cudaDeviceSynchronize();
		}

		//Cache first bounce
		if (depth == 0 && iter == 1) {
			cudaMemcpy(dev_intersections_cached, dev_intersections, sizeof(ShadeableIntersection) * start_paths, cudaMemcpyDeviceToDevice);
		}

		//Sort by material on the first pass
		if (sortByMat && depth == 0) {
			thrust::sort_by_key(dev_thrust_intersections, dev_thrust_intersections + num_paths, dev_thrust_paths, material_id_comparator());
		}

		//Compute path shading
		if (cacheFirstBounce && depth == 0) {
			shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth, traceDepth, iter, start_paths, dev_intersections_cached, dev_paths_cached, dev_paths, dev_materials);
			checkCUDAError("shadeMaterial failed!");
		}
		else {
			shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth, traceDepth, iter, num_paths, dev_intersections, dev_paths, dev_paths, dev_materials);
			checkCUDAError("shadeMaterial failed!");
		}

		//Compact paths
		num_paths = thrust::partition(dev_thrust_paths, dev_thrust_paths + num_paths, is_not_terminated()) - dev_thrust_paths;

		depth++;
		if (num_paths == 0) {
			iterationComplete = true;
		}
	}

	if (antiAliasing == AA || antiAliasing == ADAPTIVE) {
		thrust::sort(dev_thrust_paths, dev_thrust_paths + start_paths, path_id_comparator());
	}

	// Assemble this iteration and apply it to the image
	finalGather << <blocksPerGrid2d, blockSize2d >> >(cam.resolution, dev_image, dev_paths, antiAliasing, dev_image_edges);

	///////////////////////////////////////////////////////////////////////////

	//Process image data
	processImage << <blocksPerGrid2d, blockSize2d >> >(cam.resolution, iter, dev_image, dev_image_result, antiAliasing, dev_image_edges);

	// Send results to OpenGL buffer for rendering
	if (viewEdges) {
		sendImageToPBO << <numBlocksPixels, blockSize1d >> >(pbo, pixelcount, dev_image_edges);
	}
	else {
		sendImageToPBO << <numBlocksPixels, blockSize1d >> >(pbo, pixelcount, dev_image_result);
	}

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image_result,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
